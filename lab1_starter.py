import random

import taichi as ti
import random as rnd

import playsound as ps

ti.init(arch=ti.gpu)

#Parameteros pieza de tela 1x1
n = 100 #Numero de particulas en cada direccion
cell_size = 1.0 / (n - 1) #Distancia entre particulas
cell_mass = 1.0 / (n * n)

cloth_corner = ti.Vector([-0.5, 0.7, -0.5])

cloth_texture_path = "pirate_flag.png"

#Posiciones de las particulas
x = ti.Vector.field(3, float, shape=(n,n))
v = ti.Vector.field(3, float, shape=(n,n))
a = ti.Vector.field(3, float, shape=(n,n))

vertices = ti.Vector.field(3, float, shape=n*n)
triangles = (n-1) * (n-1) * 2
indexes = ti.field(int, shape=triangles * 3)
colors = ti.Vector.field(3, float, shape=n*n)

#Parametros simulacion
frame_dt = 0.01
substeps = 150
dt = frame_dt / substeps
g = ti.Vector([0, -9.81, 0])
k = 200 #Coeficiente del muelle
damp_c = 0.2 #Coeficiente de amortiguamiento
air_c = 0.0001 #Coeficiente de rozamiento
wind_min = 0
wind_max = 0.5
wind_dir_max_angle = 10
wind_dir = ti.Vector([1.0, 0.1, 0.1]).normalized()

box_vertices = ti.Vector.field(3, dtype=ti.f32, shape=8)
box_indices = ti.field(dtype=ti.i32, shape=36)

sx, sy, sz = 10.0, 100.0, 10.0  # cubo estirado: poste fino y alto
cx, cy, cz = cloth_corner     # centro

@ti.kernel
def build_box():
    hx, hy, hz = sx * 0.5, sy * 0.5, sz * 0.5

    box_vertices[0] = [cx - hx, cy - hy, cz - hz]
    box_vertices[1] = [cx + hx, cy - hy, cz - hz]
    box_vertices[2] = [cx + hx, cy + hy, cz - hz]
    box_vertices[3] = [cx - hx, cy + hy, cz - hz]

    box_vertices[4] = [cx - hx, cy - hy, cz + hz]
    box_vertices[5] = [cx + hx, cy - hy, cz + hz]
    box_vertices[6] = [cx + hx, cy + hy, cz + hz]
    box_vertices[7] = [cx - hx, cy + hy, cz + hz]

    idx = [
        0, 1, 2,  0, 2, 3,   # cara trasera
        4, 6, 5,  4, 7, 6,   # cara delantera
        0, 4, 5,  0, 5, 1,   # abajo
        3, 2, 6,  3, 6, 7,   # arriba
        0, 3, 7,  0, 7, 4,   # izquierda
        1, 5, 6,  1, 6, 2    # derecha
    ]

    for i in ti.static(range(36)):
        box_indices[i] = idx[i]

@ti.kernel
def generate_colors():
    for i, j in x: #ti.ndrange(n, n)
        if (i // 4 + j // 4) % 2:
            colors[i * n + j] = 1.0
        else:
            colors[i * n + j] = (0.0, 0.4, 0.94)

@ti.kernel
def generate_colors_from_texture(texture: ti.types.ndarray(dtype=float, ndim=3)):
    for i, j, _ in texture:
        colors[i * n + j] = texture[i, j, 0]

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
    rand = (ti.random(dtype=float) * 2 - 1) * wind_dir_max_angle
    rand_vec = ti.Vector([rand, rand, rand])
    rand_dir = (wind_dir + rand_vec)
    F_viento = rand_dir * ((ti.random(dtype=float) * wind_max) + wind_min) / substeps
    for I in ti.grouped(x):
        #Fuerzas (Recordar sumar las fuerzas pequeñas primero y las grandes después.)
        F_gravity = cell_mass * g

        F_amortiguamiento = ti.Vector([0.0, 0.0, 0.0])

        #Fuerza muelles estructurales (+ amortiguamiento)
        F_estructural = ti.Vector([0.0, 0.0, 0.0])
        for O in ti.static([(-1,0), (1,0), (0,-1), (0,1)]):
            J = I + O
            if 0 <= J[0] < n and 0 <= J[1] < n:
                Pij = x[J] - x[I]
                if Pij.norm() > 1e-6:
                    Vij = v[I] - v[J]
                    F_estructural += k * (Pij.norm() - cell_size) * Pij.normalized()
                    F_amortiguamiento += -damp_c * (Vij.dot(Pij.normalized())) * Pij.normalized()

        # Fuerza muelles deformacion (+ amortiguamiento)
        F_deformacion = ti.Vector([0.0, 0.0, 0.0])
        for O in ti.static([(-1, -1), (-1, 1), (1, -1), (1, 1)]):
            J = I + O
            if 0 <= J[0] < n and 0 <= J[1] < n:
                Pij = x[J] - x[I]
                if Pij.norm() > 1e-6:
                    Vij = v[I] - v[J]
                    L = ti.sqrt((cell_size ** 2) + (cell_size ** 2))
                    F_deformacion += k * (Pij.norm() - L) * Pij.normalized()
                    F_amortiguamiento += -damp_c * (Vij.dot(Pij.normalized())) * Pij.normalized()

        # Fuerza muelles flexion (+ amortiguamiento)
        F_flexion = ti.Vector([0.0, 0.0, 0.0])
        for O in ti.static([(-2,0), (2,0), (0,-2), (0,2)]):
            J = I + O
            if 0 <= J[0] < n and 0 <= J[1] < n :
                Pij = x[J] - x[I]
                if Pij.norm() > 1e-6:
                    Vij = v[I] - v[J]
                    F_flexion += k * (Pij.norm() - cell_size * 2) * Pij.normalized()
                    F_amortiguamiento += -damp_c * (Vij.dot(Pij.normalized())) * Pij.normalized()

        #Fuerza de rozamiento
        F_aire = -air_c * v[I]
        F = F_gravity + F_estructural + F_deformacion + F_flexion + F_amortiguamiento + F_aire + F_viento
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
            cloth_corner.y + j * cell_size,
            cloth_corner.z,
        ]

        v[i, j] = [0,0,0]

def init_texture():
    texture = ti.tools.imread(cloth_texture_path)
    texture = ti.tools.imresize(texture, n, n) / 255

    ti_texture = ti.ndarray(dtype=float, shape=(n, n, 3))
    ti_texture.from_numpy(texture)

    return ti_texture

ball_center = ti.Vector.field(3, dtype=float, shape=1)

ball_radius = 0.3
ball_color = (0.5, 0.42, 0.8)

def init():
    ball_center[0] = ti.Vector([0, 0, 0])
    init_cloth()
    calculate_indexes()
    generate_colors_from_texture(init_texture())

def main():
    window = ti.ui.Window("Telita chuli", (512, 512), fps_limit=60)
    canvas = window.get_canvas()
    scene = window.get_scene()

    canvas.set_background_color((0.4, 0.74, 0.85))

    camera = ti.ui.Camera()
    camera.position(0.0, 1.0, 1.5)
    camera.lookat(0.0, 1.0, 0.0)
    scene.set_camera(camera)

    init()

    ps.playsound("pirate_sound.mp3", block=False)

    while window.running:
        #Simulacion
        for _ in range(substeps):
            step()

        #Renderizado
        update_vertices()
        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(camera)

        scene.point_light(ball_center[0], (1., 1., 1.)) # (0., 1., 2.)
        scene.ambient_light((0.5, 0.5, 0.5))

        #scene.particles(ball_center, radius=ball_radius, color=ball_color)
        scene.mesh(vertices, indexes, per_vertex_color=colors, show_wireframe=False)
        scene.mesh(box_vertices, box_indices, color=(0.14, 0.59, 0.75), show_wireframe=True)

        canvas.scene(scene)
        window.show()

if __name__ == "__main__":
    main()