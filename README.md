# optical_tracer

Optical rays tracer. Domain core consists of OpticalSystem cls which has name, components and rays public properties.  Components are opticals which are compose the optical system and could be added with add_component() method. Rays are light traces and could be calculated with trace(). Initial vector of each ray could be add with add_initial_vector().
Anyway optical system could be built with OpticalSystemBuilder cls:
In the core are defined such classes like Side and Boundary which are needed to compose a Layer - a curve of defined form with given side relative to the boundary. Component is an agregation of layers, for example two circular curves with right and left sides and given Material properties. A many of components composes optical system.
Use create_side(), create_boundary_callable(), create_material(), create_layer(), create_component(), add_components() to compose an optical system.
Use create_point(), create_vector() to create vectors to trace. Add vectors to OpticalSystemBuilder().vectors.
Use trace_all() to trace vectors through the composed optical system.
Coordinates of rays could be found in self.vectors as a list of node vectors. Each node is ray's refraction or reflaction point.

Added web representation. Django is used. Formes+bootstrap handling added to create and Optical systems. 

Ideas:
Tkinter representation.
