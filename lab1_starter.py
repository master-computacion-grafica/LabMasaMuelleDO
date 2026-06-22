import taichi as ti

ti.init(arch=ti.gpu)

#Parameteros pieza de tela 1x1
n = 15 #Numero de particulas en cada direccion
cell_size = 1.0 / (n - 1) #Distancia entre particulas
cell_mass = 1.0 / (n * n)

cloth_corner = ti.Vector([-0.5, 0.7, -0.5])

#Posiciones de las particulas
x = ti.Vector.field(3, float, shape=(n,n))
v = ti.Vector.field(3, float, shape=(n,n))
a = ti.Vector.field(3, float, shape=(n,n))

vertices = ti.Vector.field(3, float, shape=n*n)
triangles = (n-1) * (n-1) * 2
indexes = ti.field(int, shape=triangles * 3)
colors = ti.Vector.field(3, float, shape=n*n)

#Parametros simulacion
dt = 0.01
g = ti.Vector([0, -9.81, 0])
k = 10

@ti.kernel
def generate_colors():
    for i, j in x: #ti.ndrange(n, n)
        if (i // 4 + j // 4) % 2:
            colors[i * n + j] = 1.0
        else:
            colors[i * n + j] = (0.0, 0.4, 0.94)
@ti.kernel
def calculate_indexes():
    for i, j in ti.ndrange(n-1, n-1): #Recorro los cuadrados
        quad_id = i * (n - 1) + j
        #Rellernar indices primer triangulo
        indexes[quad_id * 6 + 0] = i * n + j
        indexes[quad_id * 6 + 1] = (i + 1) * n + j
        indexes[quad_id * 6 + 2] = i * n + j + 1
        # Rellernar indices segundo triangulo
        indexes[quad_id * 6 + 3] = i * n + j + 1
        indexes[quad_id * 6 + 4] = (i + 1) * n + j
        indexes[quad_id * 6 + 5] = (i + 1) * n + j + 1
@ti.kernel
def step(): # Symplectic Euler
    for I in ti.grouped(x):
        #Fuerzas (Recordar sumar las fuerzas pequeñas primero y las grandes después.)
        F_gravity = cell_mass * g

        F_amortiguamiento = ti.Vector([0.0, 0.0, 0.0])


        #Fuerza muelles estructurales (+ amortiguamiento)
        F_estructural = ti.Vector([0.0, 0.0, 0.0])
        for O in  ti.static([(-1,0), (1,0), (0,-1), (0,1)]):
            J = I + O
            if 0 <= J[0] < n and 0 <= J[1] < n:
                Pij = x[J] - x[I]
                Pji = x[I] - x[J]
                F_estructural += k * (Pij.norm() - cell_size) * Pij.normalized()
                F_estructural -= k * (Pji.norm() - cell_size) * Pji.normalized()


        F = F_gravity + F_estructural
        #Aceleracion
        a[I] = F / cell_mass

    for I in ti.grouped(x):
        #Actualizacion de la velocidad y la posicion
        v[I] += a[I]*dt

    v[0, 0] = [0, 0, 0]
    v[0, n - 1] = [0, 0, 0]

    for i, j in x:
        x[i, j] += v[i, j] * dt

@ti.kernel
def update_vertices():
    for i, j in x:
        vertices[i*n + j] = x[i, j]

@ti.kernel
def init_cloth():
    for i, j in x:
        x[i, j] = [
            cloth_corner.x + i * cell_size,
            cloth_corner.y,
            cloth_corner.z + j * cell_size,
        ]

        v[i, j] = [0,0,0]

ball_center = ti.Vector.field(3, dtype=float, shape=1)

ball_radius = 0.3
ball_color = (0.5, 0.42, 0.8)


def init():
    ball_center[0] = ti.Vector([0, 0, 0])
    init_cloth()
    calculate_indexes()
    generate_colors()


def main():
    window = ti.ui.Window("Bola 3D", (512, 512), fps_limit=60)
    canvas = window.get_canvas()
    scene = window.get_scene()

    camera = ti.ui.Camera()
    camera.position(0.0, 0.0, 3.0)
    camera.lookat(0.0, 0.0, 0.0)
    scene.set_camera(camera)

    init()

    while window.running:
        #Simulacion
        step()

        #Renderizado
        update_vertices()
        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(camera)

        scene.point_light((0., 1., 2.), (1., 1., 1.))
        scene.ambient_light((0.5, 0.5, 0.5))

        scene.particles(ball_center, radius=ball_radius, color=ball_color)
        scene.mesh(vertices, indexes, per_vertex_color=colors, show_wireframe=False)

        canvas.scene(scene)
        window.show()


if __name__ == "__main__":
    main()
