import taichi as ti
import os, math, sys

ti.init(arch=ti.gpu)
vec = ti.math.vec2

SAVE_FRAMES = True

window_size = 1024  # Number of pixels of the window
n = 2500  # Number of grains
MAX_PARTICLES = n * 2

density = 1500.0
stiffness = 8e3
restitution_coef = 0.3  # how “bouncy” collisions are, i.e. how much kinetic energy is conserved when two particles (or a particle and a wall) collide.

#A small restitution_coef (e.g. 0.001) → strong damping → nearly completely inelastic collisions → grains lose vertical velocity quickly and settle.
#A large restitution_coef (e.g. 0.8) → weak damping → grains bounce several times before resting → more “energetic” behavior.

gravity = -9.81
dt = 0.0001  # Larger dt might lead to unstable results.
substeps =90
h_dem = dt/substeps

# PBD Parameters
pbd_dt =  1./1024.
pbd_substeps = 12
h_pbd = pbd_dt / pbd_substeps
h_pbd_sqr = h_pbd * h_pbd
inv_time_step = 1.0 / h_pbd
C_W_ACC = 1.01

MU_S = 100.99
MU_K = 0.5


@ti.dataclass
class Grain:
    p: vec  # Position
    m: ti.f32  # Mass
    inv_m: ti.f32  # Inverse Mass
    r: ti.f32  # Radius
    v: vec  # Velocity
    a: vec  # Acceleration
    f: vec  # Force
    active: ti.i32
    color_index: ti.i32
    prev: vec  # Previous Position
    dp: vec # delta p for pbd jacobi 
    dp_ctr: ti.i32 # counter for PBD jacobi 
    dv: vec
    dv_ctr: ti.i32

gf = Grain.field(shape=(MAX_PARTICLES, ))
n_particles = ti.field(dtype=ti.i32, shape=())

y_floor = ti.field(dtype=ti.f32, shape=())
C_FREQUENCY = 0.0003;
C_FREQUENCY_PBD =  0.25;

# steps in 1 frame / sec:  substeps * C_FREQUENCY / dt

C_FREQUENCY_PBD= (substeps * C_FREQUENCY / dt) * pbd_dt / pbd_substeps

C_AMPLITUDE = 0.0#0.021; # good baseline is 0.02, but 0.019 also works



grain_r_min = 0.002
grain_r_max = 0.003

GRAV = vec(0., gravity )
# -------------------------------------
# GRID STUFF
# Grid Parameters

#grid_n = 128
#grid_size = 1.0 / grid_n  # Simulation domain of size [0, 1]

cell_size = grain_r_max * 6
neighbor_radius = grain_r_max * 1.05 * 5
# TODO, debug specifc bug with neighbor_radius = MAX_R * 1.05 * 9?
cell_recpr = 1.0 / cell_size

def round_up(f, s):
    return (math.floor(f * cell_recpr / s) + 1) * s

grid_num_particles = ti.field(int)
particle_neighbors = ti.field(int)
error_flag = ti.field(dtype=ti.i32, shape=())

max_num_particles_per_cell = 50
max_num_neighbors = max_num_particles_per_cell*5 

grid_start = ti.Vector([0.0,0.0])
boundary = (  1.0,  1.0 )
grid_size = (round_up(boundary[0], 1), 
            round_up(boundary[1], 1))

grid_snode = ti.root.dense(ti.ij, grid_size)
grid_snode.place(grid_num_particles)
grid2particles = ti.field(int, (grid_size + (max_num_particles_per_cell,)))
nb_node = ti.root.dense(ti.i, MAX_PARTICLES)
particle_num_neighbors = ti.field(int)
nb_node.place(particle_num_neighbors)
nb_node.dense(ti.j, max_num_neighbors).place(particle_neighbors)

moving_right_wall = ti.field(ti.f32, shape=())
alpha_slope = ti.field(ti.f32, shape=())

@ti.func
def is_in_grid(c):
    # @c: Vector(i32)
    return 0 <= c[0] and c[0] < grid_size[0] and 0 <= c[1] and c[
        1] < grid_size[1] 

@ti.func
def get_cell(pos):
    # Grid starts at grid_start
    q = ( (pos - grid_start) * cell_recpr)
    return ti.cast(q, ti.i32)   # component-wise cast

# ------------------------------------------
# end grid code
# ------------------------------------------
# velocity pass for PBD
MAX_CONS = n* 11
cons_size = ti.field(ti.i32, shape=())
cons_a = ti.field(ti.i32, shape=MAX_CONS)
cons_b = ti.field(ti.i32, shape=MAX_CONS)
cons_dv = ti.Vector.field(2, ti.f32, shape=MAX_CONS)
cons_n = ti.Vector.field(2, ti.f32, shape=MAX_CONS)
cons_dlam = ti.field(ti.f32, shape=MAX_CONS)


@ti.kernel
def deactivate_right(x_th: ti.f32):
    for i in range(n_particles[None]):
        if gf[i].p[0] > x_th:
            gf[i].active = 0

@ti.kernel
def init():
    # Sand column collapse initial condition: rectangular column resting on the ground.
    # Pack grains in a grid inside [x0, x0+W] x [0, H].
    x0 = grain_r_max
    W = 0.98
    H = 0.32
    moving_right_wall[None] = 0.3
    alpha_slope[None] = math.pi/4.0
    r0 = (grain_r_min + grain_r_max) * 0.5
    y0 = r0
    s = 2.01 * r0  # spacing > 2r to avoid overlap
    cols = max(0.3, int(0.3 / s)) #max(1, int(W / s))
    rows = max(1, int(H / s))
    n_particles[None] = min(cols * rows, MAX_PARTICLES)
    y_floor[None] = 0.01
    error_flag[None] = 0
    for i in gf:
        if i < n_particles[None]:
            ix = i % cols
            iy = i // cols
            px = x0 + (ix + 0.5) * s + ti.random()*grain_r_min
            py = y0 + (iy + 0.5) * s+ ti.random()*grain_r_min
            gf[i].p = vec(px, py)
            gf[i].r = ti.random() * (grain_r_max - grain_r_min) + grain_r_min
            gf[i].v = vec(0.0, 0.0)
            gf[i].a = vec(0.0, 0.0)
            gf[i].f = vec(0.0, 0.0)
            gf[i].active = 1
            gf[i].color_index = -1
            gf[i].m = density * math.pi * gf[i].r**2
            gf[i].inv_m = 1.0/gf[i].m

@ti.kernel
def add_partilces()->int:
    base = n
    added = 0
    padding = 0.32
    region_width = 0.8 - padding * 2
    for ix in range(-16, 17):
        for iy in range(-16, 17):
            idx = base + added
            if idx < MAX_PARTICLES:                
                l = added * grid_size
                pos = vec(l % region_width/3 + padding + 
                        grid_size * ti.random() * grain_r_min,
                        l // region_width/2 * grid_size + 0.01)
                gf[idx].p = pos
                gf[idx].r = ti.random() * (grain_r_max - grain_r_min) + grain_r_min
                gf[idx].m = density * math.pi * gf[idx].r**2
                added += 1
    n_particles[None] += added
    return added 

@ti.kernel
def update(angle: float):
    y_floor[None] = C_AMPLITUDE * ( 1.0+  ti.sin(angle))
    for i in gf:
        if i < n_particles[None] and gf[i].active > 0:
            a = gf[i].f / gf[i].m
            # Linear
            gf[i].v += (gf[i].a + a) * h_dem
            gf[i].p += gf[i].v * dt
            gf[i].a = a


@ti.kernel
def update_pbd(angle: float):
    y_floor[None] = C_AMPLITUDE * ( 1.0+  ti.sin(angle))
    cons_size[None] = 0
    for i in gf:
        if i < n_particles[None] and gf[i].active > 0:
            gf[i].prev = gf[i].p
            #gf[i].a = vec(0., gravity) 
            #gf[i].v += (gf[i].a + a) * dt
            gf[i].v += GRAV * h_pbd
            gf[i].p += gf[i].v * h_pbd   
            gf[i].dp.x = 0.0
            gf[i].dp.y = 0.0
            gf[i].dp_ctr = 0
            gf[i].dv.x = 0.0
            gf[i].dv.y = 0.0
            gf[i].dv_ctr = 0


@ti.kernel
def apply_bc():
    # also applies friction
    bounce_coef = 0.7  # Velocity damping
    for i in gf:
        gf[i].f = vec(0., gravity * gf[i].m)  # Apply gravity.

        if i < n_particles[None] and gf[i].active > 0:
            x = gf[i].p[0]
            y = gf[i].p[1]

            if y - gf[i].r < y_floor[None]:
                gf[i].p[1] = y_floor[None] + gf[i].r
                gf[i].v[1] *= -bounce_coef
                gf[i].v[0] *=0.1 

            elif y + gf[i].r > 1.0:
                gf[i].p[1] = 1.0 - gf[i].r
                gf[i].v[1] *= -bounce_coef


            if x - gf[i].r < 0:
                gf[i].p[0] = gf[i].r
                gf[i].v[0] *= -bounce_coef

            elif x + gf[i].r > moving_right_wall[None]:
                gf[i].p[0] = moving_right_wall[None] - ( 0.01 * ti.random() + 1.0) * gf[i].r 
                gf[i].v[0] *= -bounce_coef




@ti.func
def calc_friction(path, n, dist):
    # JS behavior: if velocity pass is enabled, only static friction in position solve.
    res =  vec(0.0, 0.0)
    proj = path.dot(n)
    vt = path - proj * n
    vt_len = vt.norm()
    if vt_len < MU_S * dist:
        res=vt
    else:
        k = 0.0
        if MU_K != 0.0 and vt_len > 1e-9:
            k = ti.min(MU_K * dist / vt_len, 1.0)
        res= k * vt
    return res


@ti.kernel
def solve_sphere_sphere():
    for i in gf:
        if i < n_particles[None] and gf[i].active > 0:
            for n_j in range(particle_num_neighbors[i]):
                j = particle_neighbors[i, n_j]
                if j >= 0 and i<j:
                    pi = gf[i].p
                    pj = gf[j].p
                    d = pi - pj
                    dist = d.norm()
                    # normal
                    nrm = vec(0.0, 0.0)
                    if dist > 1e-9:
                        nrm = d / dist
                    C = (gf[i].r + gf[j].r ) - dist
                    if C > 1e-5:
                        joint_inv_m = gf[i].inv_m + gf[j].inv_m 
                        coef_i =  gf[i].inv_m / joint_inv_m 
                        corr_i = coef_i  * C * nrm  # equal mass
                        coef_j =  gf[j].inv_m / joint_inv_m 
                        corr_j = coef_j* C * nrm  # equal mass
                        # relative path (for friction + constraint)
                        path = (gf[i].p + corr_i - gf[i].prev) - (gf[j].p - corr_j - gf[j].prev )
                        fr = calc_friction(path, nrm, C)
                        fr_i = - coef_i * fr
                        fr_j = coef_j * fr
                        ti.atomic_add( gf[i].dp, corr_i + fr_i)
                        ti.atomic_add( gf[j].dp,- corr_j + fr_j)
                        ti.atomic_add(gf[i].dp_ctr ,1)
                        ti.atomic_add(gf[j].dp_ctr ,1)


@ti.func
def resolve(i, j):
    rel_pos = gf[j].p - gf[i].p
    dist = ti.sqrt(rel_pos[0]**2 + rel_pos[1]**2)
    delta = -dist + gf[i].r + gf[j].r  # delta = d - 2 * r
    if delta > 0:  # in contact
        normal = rel_pos / dist
        f1 = normal * delta * stiffness
        # Damping force
        M = (gf[i].m * gf[j].m) / (gf[i].m + gf[j].m)
        K = stiffness
        C = 2. * (1. / ti.sqrt(1. + (math.pi / ti.log(restitution_coef))**2)
                  ) * ti.sqrt(K * M)
        V_relative = (gf[j].v - gf[i].v) 
        V = V_relative * normal
        zeta = V_relative.dot(normal)
        V_tangential = V_relative - zeta * normal
        V_tangential_mag = V_tangential.norm()
        f2 = C * V * normal
        contact_normal_mag = (f2 - f1).norm()
        ti.atomic_add(gf[i].f,f2 - f1 )
        ti.atomic_add(gf[j].f,-f2 +f1 )
        if(V_tangential_mag > 0.0001):
            V_tangential_norm = V_tangential / V_tangential_mag
            # Tangential friction force (existing):
            Ft_vec = ti.tan(alpha_slope[None]) * contact_normal_mag * V_tangential_norm
            ti.atomic_add(gf[i].f,Ft_vec )
            ti.atomic_add(gf[j].f,-Ft_vec )



@ti.kernel
def stability_grid(gf: ti.template()):
    # clear neighbor lookup table
    for I in ti.grouped(grid_num_particles):
        grid_num_particles[I] = 0
    for I in ti.grouped(particle_neighbors):
        particle_neighbors[I] = -1

    # update grid
    max_num_in_cell =  ti.cast(0, ti.i32)
    for i in gf:
        if i < n_particles[None] and gf[i].active > 0:
            cell = get_cell( gf[i].p )
            # ti.Vector doesn't seem to support unpacking yet
            # but we can directly use int Vectors as indices
            offs = ti.atomic_add(grid_num_particles[cell], 1)
            grid2particles[cell, offs] = i

    for I in ti.grouped(grid_num_particles):
        max_num_in_cell= ti.max(max_num_in_cell, grid_num_particles[I])

    for p_i in gf:
        if p_i < n_particles[None] and gf[p_i].active > 0:
            pos_i = gf[p_i].p
            cell = get_cell( pos_i )
            nb_i = 0
            for offs in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2)))):
                cell_to_check = cell + offs
                if is_in_grid(cell_to_check):
                    for j in range(grid_num_particles[cell_to_check]):
                        p_j = grid2particles[cell_to_check, j]
                        pos_j = gf[p_j].p
                        if gf[p_j].active > 0 and nb_i < max_num_neighbors and p_j != p_i and (
                                pos_i - pos_j).norm() < neighbor_radius:
                            particle_neighbors[p_i, nb_i] = p_j
                            nb_i += 1
            if nb_i >= max_num_neighbors:
                print('over max number of neigbhours',nb_i, max_num_neighbors)
                error_flag[None] = 1
            particle_num_neighbors[p_i] = nb_i



@ti.kernel
def contact(gf: ti.template()):
    '''
    Handle the collision between grains.
    '''
    for i in gf:
        if i < n_particles[None] and gf[i].active > 0:
            for n_j in range(particle_num_neighbors[i]):
                j = particle_neighbors[i, n_j]
                if j >= 0 and gf[j].active > 0 and i<j:
                    resolve(i, j)



@ti.kernel
def update_vel_from_pos():
    for i in gf:
        if i < n_particles[None] and gf[i].active > 0:
            if gf[i].dp_ctr > 0:
                #pos_addons = ((C_W_ACC * dp[i]) / ti.cast(ctr, ti.f32) ).norm()
                #ti.atomic_add(total_sum, overlap_metrics[i] /  ti.cast(ctr, ti.f32))
                #ti.atomic_add(total_sum_ctr, 1)
                gf[i].p += (C_W_ACC * gf[i].dp) / ti.cast(gf[i].dp_ctr , ti.f32)
                gf[i].dp = vec(0.0, 0.0)
                gf[i].dp_ctr = 0
            gf[i].v = (gf[i].p - gf[i].prev) * inv_time_step
            gf[i].v *= 0.999 # apply damping


@ti.kernel
def apply_dp():
    for i in gf:
        if i < n_particles[None] and gf[i].active > 0 and gf[i].dp_ctr > 0:
            gf[i].p += (C_W_ACC * gf[i].dp) / ti.cast(gf[i].dp_ctr , ti.f32)
            gf[i].dp = vec(0.0, 0.0)
            gf[i].dp_ctr = 0
            
@ti.kernel
def assign_color_indicies(num_colors: int):
    max_height= -999.0
    min_height = 999.0
    for i in gf:
        if i < n_particles[None] and gf[i].active > 0:
            y = gf[i].p[1]
            min_height = ti.min(y, min_height)
            max_height = ti.max(y, max_height)
    range_ = max_height-min_height

    for i in gf:
        if i < n_particles[None] and gf[i].active > 0:
            y = gf[i].p[1]
            gf[i].color_index = ti.floor( num_colors * (y - min_height) / range_)
        else:
            gf[i].color_index  = 0

def pbd_simulation(update_no, frame):
    stability_grid(gf)
    for _ in range(pbd_substeps):
        angle = update_no* C_FREQUENCY_PBD
        update_pbd(angle)
        solve_sphere_sphere()
        update_vel_from_pos()
        apply_bc()
    update_no+=1
    return update_no

def dem_simulation(update_no):
    stability_grid(gf)
    for s in range(substeps):
        angle = update_no* C_FREQUENCY
        update(angle)
        apply_bc()
        contact(gf)
        update_no+=1
    return update_no











if __name__ == "__main__":
    pbd_mode = False
    if len(sys.argv) >=2:
        if sys.argv[1] == "--pbd":
            pbd_mode = True
            print("Starting PBD sand simulation")
        else:
            print("Starting DEM simulation")
    else:
        print("Starting DEM simulation")

    init()
    gui = ti.GUI('Taichi DEM', (window_size, window_size))
    step = 0

    deactivated = False
    if SAVE_FRAMES:
        os.makedirs('output', exist_ok=True)

    update_no=0
    frame=0
    mouse_coordinates = (0.3, 1.0)

    mu_tangential = gui.slider('Angle of Repose', 0, math.pi/4.0, step=0.05)
    while gui.running:
        for e in gui.get_events(gui.PRESS):
            if e.key == gui.ESCAPE:
                gui.running = False
            elif e.key == 'a':
                print("hello")
                added = add_partilces()
                n+=added
                print(added,n)
            mouse_coordinates = gui.get_cursor_pos()
            print(mouse_coordinates)
        #print(mu_tangential.value)
        alpha_slope[None] = mu_tangential.value
        moving_right_wall[None] =  (300*moving_right_wall[None] + mouse_coordinates[0])/301.0
        if pbd_mode:
            update_no = pbd_simulation(update_no, frame)
        else:
            update_no = dem_simulation(update_no) 
        frame+=1
        if frame % 50 == 0:
            ti.sync()  # IMPORTANT on GPU
            #ti.profiler.print_kernel_profiler_info()
            #ti.profiler.clear_kernel_profiler_info()    
        if error_flag[None] != 0:
            raise RuntimeError("Taichi kernel detected invalid state") 
        pos = gf.p.to_numpy()

        r = gf.r.to_numpy() * window_size
        gui.circles(pos, radius=r )

        # for drawing mouse, wall position and so on 
        gui.circle(mouse_coordinates, radius=10.0)
        gui.rect(
            (0,  y_floor[None]),
            (moving_right_wall[None], 1.0),
            color=0xFFFF00,
        )    
        gui.rect(
            (0,  y_floor[None]),
            (1.0, 0),
            color=0xFFFFFF,
        )
        if SAVE_FRAMES:
            gui.show(f'output/{step:06d}.png')
        else:
            gui.show()

        step += 1
