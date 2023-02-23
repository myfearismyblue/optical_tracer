__all__ = ['FormHandleBaseStrategy',
           'fetch_optical_system_by_id',
           'AddComponentFormHandleService',
           'ChooseOpticalSystemFormHandleService']

from abc import ABC, abstractmethod
from typing import Optional, Dict, Callable
from warnings import warn

import dill
from django.forms import forms

from core import IOpticalSystem, OpticalSystem, UnspecifiedFieldException, OpticalComponent
from ._infrastructural_exceptions import WrongBoundaryEquationSyntaxError, EmptyBoundaryEquationSyntaxError
from ._context_registry import ContextRegistry
from ._optsys_builder import OpticalSystemBuilder, IOpticalSystemBuilder
from ._context_requests import ContextRequest
from ..forms import AddComponentForm, ChooseOpticalSystemForm
from ..models import OpticalSystemView


class FormHandleBaseStrategy(ABC):
    """
    Abstract cls to perform different form handle strategies. Each strategy gets certain form,
    performs handling of certain data type and returns name of a context in ContextRequest which is to be given to graph
    service in order to prepare this context"""

    @property
    def optical_system_id(self):
        return self._optical_system_id

    @optical_system_id.setter
    def optical_system_id(self, opt_sys_id):
        if not isinstance(opt_sys_id, (int, type(None))):  # for 3.0<=python<=3.9
            raise TypeError(f'Wrong id for inner optical system in FormHandleService: {opt_sys_id}')
        opt_sys_id = (None if opt_sys_id is None else int(opt_sys_id))
        self._optical_system_id = opt_sys_id

    @abstractmethod
    def handle(self, form_instance: forms.Form) -> ContextRequest:
        ...


def fetch_optical_system_by_id(*, id: Optional[int]) -> IOpticalSystem:
    """
    Returns optical system fetching it from db using given id as pk.
    If pk is None or invalid returns an empty optical system
    """
    if id is None or not isinstance(id, int):
        warn(f'Fetching optical system is unsuccessful. Given type(id): {type(id)}. '
             f'An empty optical system has been created')
        return OpticalSystemBuilder().optical_system
    try:
        modelOpticalSystemView = OpticalSystemView.objects.get(pk=id)
    except OpticalSystemView.DoesNotExist:
        warn(f'Optical system with the given pk is not found. pk={id}. An empty optical system has been created')
        return OpticalSystemBuilder().optical_system

    optical_system = dill.loads(modelOpticalSystemView.opt_sys_serial)
    assert isinstance(optical_system, OpticalSystem), f'Fetched from DB: {optical_system}'
    return optical_system


class AddComponentFormHandleService(FormHandleBaseStrategy):
    """Responsible for handling django form AddComponentForm and forwarding it to domain model.
    An optical system pk should be give while instance the cls, in which component is going to be added.
    If opt sys is not given, the new one will be created"""

    def __init__(self, *, opt_sys_id: Optional[int] = None) -> None:
        self.optical_system_id = opt_sys_id
        optical_system: IOpticalSystem = fetch_optical_system_by_id(
            id=opt_sys_id)  # empty optical system if id is invalid
        self._builder: IOpticalSystemBuilder = OpticalSystemBuilder(optical_system=optical_system)

    def handle(self, form_instance: AddComponentForm):
        if not isinstance(form_instance, AddComponentForm):
            raise TypeError(f'Wrong type of argument for this type of handler.'
                            f'Should be AddComponentForm form, but was given {type(form_instance)}')
        if form_instance.is_valid():
            self._create_and_pull_new_component(form_instance.cleaned_data)

        return ContextRegistry().get_context_name('opt_sys_context')

    @property
    def builder(self) -> IOpticalSystemBuilder:
        """Cls uses OpticalSystemBuilder to handle with domain model"""
        if not isinstance(self._builder, IOpticalSystemBuilder):
            raise UnspecifiedFieldException(f'Optical system builder hasn''t been initialised properly. ')
        return self._builder

    def _create_and_pull_new_component(self, cleaned_data: Dict):
        """
        Creates a new optical component via cleaned data from form.
        After creating pushes the component to builder.optical_system and traces vectors
        After that updates optical system in db
        """
        try:
            new_component: OpticalComponent = self._compose_new_component(cleaned_data)
            self.builder.add_components(components=new_component)
        except WrongBoundaryEquationSyntaxError as e:
            warn(f'WrongBoundaryEquationSyntaxError is occurred. No component is added. {e.args}')
            pass  # if any equation is wrong just do nothing with optical system
        self.builder.trace_all()
        self._update_optical_system()

    def _compose_new_component(self, cleaned_data: Dict) -> OpticalComponent:
        """Gets data from form.cleaned_data and uses OpticalSystemBuilder to compose a new OpticalComponent object."""
        layers = []  # layers to build the component
        try:
            first_layer_side = self.builder.create_side(side=str(cleaned_data['first_layer_side']))
            first_layer_boundary: Callable = self.builder.create_boundary_callable(
                equation=cleaned_data['first_layer_equation'])
            first_new_layer = self.builder.create_layer(name=cleaned_data['first_layer_name'],
                                                        side=first_layer_side,
                                                        boundary=first_layer_boundary)
            layers.append(first_new_layer)
        except EmptyBoundaryEquationSyntaxError:
            pass  # if equation is empty just do nothing

        try:
            second_layer_side = self.builder.create_side(side=str(cleaned_data['second_layer_side']))
            second_layer_boundary: Callable = self.builder.create_boundary_callable(
                equation=cleaned_data['second_layer_equation'])
            second_new_layer = self.builder.create_layer(name=cleaned_data['second_layer_name'],
                                                         side=second_layer_side,
                                                         boundary=second_layer_boundary)
            layers.append(second_new_layer)
        except EmptyBoundaryEquationSyntaxError:
            pass  # if equation is empty just do nothing with optical system

        current_material = self.builder.create_material(name=cleaned_data['material_name'],
                                                        transmittance=cleaned_data['transmittance'],
                                                        refractive_index=cleaned_data['index'])
        new_component = self.builder.create_component(name=cleaned_data['component_name'],
                                                      layers=layers,
                                                      material=current_material)
        return new_component

    def _update_optical_system(self) -> None:
        """
        Gets OpticalSystemView object with self.optical_system_id as pk.
        If not exist in db, creates an empty model object
        Serializes the current optical system which is in self.builder.optical_system.
        Sets serialized optical system and it's name to that object and saves it.
        @return: None
        """
        try:
            current_opt_sys_model = OpticalSystemView.objects.get(pk=self.optical_system_id)
        except OpticalSystemView.DoesNotExist:
            current_opt_sys_model = OpticalSystemView()

        current_opt_sys_model.opt_sys_serial = dill.dumps(self.builder.optical_system)
        current_opt_sys_model.name = self.builder.optical_system.name
        current_opt_sys_model.save()


class ChooseOpticalSystemFormHandleService(FormHandleBaseStrategy):
    """Responsible for handling form of optical system choice"""

    def __init__(self, *, opt_sys_id: Optional[int] = None) -> None:
        self.optical_system_id = opt_sys_id
        optical_system = fetch_optical_system_by_id(id=opt_sys_id)
        self._builder: IOpticalSystemBuilder = OpticalSystemBuilder(optical_system=optical_system)

    @property
    def builder(self) -> IOpticalSystemBuilder:
        """Cls uses OpticalSystemBuilder to handle with domain model"""
        if not isinstance(self._builder, IOpticalSystemBuilder):
            raise UnspecifiedFieldException(f'Optical system builder hasn''t been initialised properly. ')
        return self._builder

    def handle(self, form_instance: ChooseOpticalSystemForm):
        formChooseOpticalSystem = form_instance
        if formChooseOpticalSystem.is_valid:
            modelOpticalSystemView = formChooseOpticalSystem.cleaned_data['optical_system']
            self.optical_system_id = modelOpticalSystemView.pk
            name = modelOpticalSystemView.name
            optical_system = dill.loads(modelOpticalSystemView.opt_sys_serial)
            self.builder.reset(optical_system=optical_system)
            self.builder.set_optical_system_name(name=name)
