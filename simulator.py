'''
This file runs the simulation using
jax and diffrax, a library that contains multiple
methods to solve differential equations. Using the
methods from diffrax, this function uses its
ode solving feature to solve according to the inputs of
the user. It also calculates the quantaties to be 
displayed, which include measurements of position, velocity,
angular momentum, energy, and the classic measurements of
orbital mechanics. 
'''

#imports necessary libraries
import jax.numpy as jnp
import diffrax as dr
from scipy.constants import G
import re

#function that actually runs the simulation
def simulation_runner(parameters):

    #gets ijnputs from the ijnput array
    central_body_name, central_body_radius, central_body_mass, satellite_mass, satellite_position, satellite_velocity, time_span, time_step, solver, j2_value, drag_coefficient, cross_sectional_area, atmospheric_model, third_body_mass, third_body_position = parameters
    
    #the state vector that will be used to track ijnputs. 
    state = jnp.array(satellite_position + satellite_velocity)

    #the mass variable, central body mass times gravitional constant
    mu = G * central_body_mass

    #dictionary that defines the correct diffrax solver
    solver_map = {
    "Euler": lambda: dr.Euler(),                         
    "Heun": lambda: dr.Heun(),             
    "Midpoint": lambda: dr.Midpoint(),      
    "RK5": lambda: dr.Dopri5(),            
    "RK8": lambda: dr.Dopri8(),             
    "TSIT5": lambda: dr.Tsit5(),                        
    "DOP8": lambda: dr.Dopri8(),                        
    "Symp. Euler": lambda: dr.SemiImplicitEuler(),         
    "Rev. Heun": lambda: dr.ReversibleHeun(),           
    "Leapfrog": lambda: dr.LeapfrogMidpoint(),           
    "Imp. Euler": lambda: dr.ImplicitEuler(),            
    "KVAERNO3": lambda: dr.Kvaerno3(),                 
    "KVAERNO5": lambda: dr.Kvaerno5(),                   
    }


    #defines whether optional perturbations are on or off
    j2 = (j2_value != "off")
    drag = (drag_coefficient != "off")
    third_body = (third_body_mass != "off")

    #function that handles the equations and defines them for the differential equation
    def dynamics(time, state_vector, args):

        #splits the state vector into radius and velocity of the satellite
        r = state_vector[:3]
        v = state_vector[3:]

        #performs the basic gravitional acceleration update using newton's formula
        r_norm = jnp.linalg.norm(r)
        a = -mu * r / (r_norm**3)

        #checks if j2 is on and performs additional calculation for acceleration
        if j2:

            #defines the variables needed for j2 calculation
            x, y, z = r
            z_x = z / r_norm
            j2_factor = 1.5 * j2_value * mu * (central_body_radius**2) / (r_norm**5)

            #performs the j2 calculations for each component
            jx = j2_factor * x * ((5 * (z_x**2)) - 1)
            jy = j2_factor * y * ((5 * (z_x**2)) - 1)
            jz = j2_factor * z * ((5 * (z_x**2)) - 3)

            #updates the total acceleration
            a += jnp.array([jx, jy, jz])

        #checks if drag is on and updates
        if drag:

            #splits the atmospheric model from regex into its two sections and saves them as variables
            pattern = r"([\d\.eE+-]+)\s*\*\s*exp\(\s*-h\s*/\s*([\d\.eE+-]+)\s*\)"
            match = re.match(pattern, atmospheric_model)
            if match:
                p0 = float(match.group(1))
                H = float(match.group(2))
            else:
                raise ValueError(f"Invalid atmospheric model format: {atmospheric_model}")

            #gets the current altitude and then computes the density
            altitude = jnp.linalg.norm(r) - central_body_radius
            rho = p0 * jnp.exp(-altitude / H)

            #updates the acceleration by calculation for the drag force
            v_norm = jnp.linalg.norm(v)
            drag_accel = -0.5 * drag_coefficient * cross_sectional_area * rho * v_norm * v  / satellite_mass
            a += drag_accel

        #checks if third body perturbation is on and updates
        if third_body:
            
            #gets the position of the third body and relative vectors from satellite and central body
            r3 = jnp.array(third_body_position)
            r_s3 = r - r3 
            r_cb3 = -r3 

            #updates the acceleration according to the formula for external perturbation
            a_3rd = G * third_body_mass * ((r_cb3 / jnp.linalg.norm(r_cb3)**3) - (r_s3 / jnp.linalg.norm(r_s3)**3))
            a += a_3rd
        
        #returns the derivative of the state vecotor
        return jnp.concatenate((v, a))
    
    #gets the diffrax term as the dynamics method
    solver_term = dr.ODETerm(dynamics)

    #defines the solver based on the solver map
    solver_instance = solver_map[solver]()

    #checks to enforce fixed step size or use adaptive
    if solver in ["Heun", "RK5", "RK8", "Euler", "Symp. Euler", "Leapfrog"]:
        stepsize_control = dr.ConstantStepSize()
    else:
        stepsize_control = dr.PIDController(rtol=1e-6, atol=1e-8, dtmin=time_step*0.3, dtmax=time_step*1.5)

    #finds where to save the solver
    save = dr.SaveAt(ts=jnp.arange(0, time_span, time_step))
    
    #actually runs the solver for the simulation using the correct inputs
    solved = dr.diffeqsolve(terms=solver_term, solver=solver_instance, t0=0.0, t1=time_span, dt0=time_step, y0=state, saveat=save, stepsize_controller=stepsize_control, max_steps=1000000000)

    #gets the times and positions solved by diffrax
    states = solved.ys
    positions = states[:, :3]
    velocities = states[:, 3:]
    times = solved.ts

    #checks if any component of the positions are less than or equal to the central body radius to indicate a crash
    crash_mask = jnp.linalg.norm(positions, axis=1) <= central_body_radius
    crash_indices = jnp.where(crash_mask)[0]
    crash_index = crash_indices[0] if len(crash_indices) > 0 else None

    #if there is a crash, trims the outputs to give up till the crash. 
    if crash_index is not None:
        positions = positions[:crash_index + 1]
        velocities = velocities[:crash_index + 1]
        times = times[:crash_index + 1]
        print("crashed at ", crash_index)

    #calculates the magnitude of position and velocity
    radius = jnp.linalg.norm(positions, axis=1)
    v_magnitude = jnp.linalg.norm(velocities, axis=1)

    #calculates the angular momentum and its magnitude
    angular_momentum = jnp.cross(positions, velocities)
    angular_magnitude = jnp.linalg.norm(angular_momentum, axis=1)

    #calculates drift and percent drift of angular momentum
    initial_h = angular_magnitude[0]
    angular_drift = angular_magnitude - initial_h
    percent_angular_drift = 100 * angular_drift / initial_h

    #calculates energies 
    kinetic_energy = 0.5 * satellite_mass * (v_magnitude)**2
    potential_energy = -mu * satellite_mass / radius
    total_energy = kinetic_energy + potential_energy

    #calculates drift and percent drift of total energy
    initial_E = total_energy[0]
    energy_drift = total_energy - initial_E
    percent_energy_drift = 100 * energy_drift / initial_E

    #calculates inclination angle between angular momentum and z-axis vectors
    h_unit = angular_momentum / angular_magnitude[:, None]
    z_axis = jnp.array([0.0, 0.0, 1.0])
    inclination_cos = jnp.clip(jnp.sum(h_unit * z_axis, axis=1), -1.0, 1.0)
    inclination_rad = jnp.arccos(inclination_cos)
    inclination_deg = jnp.degrees(inclination_rad)

    #calculates semi major axis
    specific_energy = 0.5 * v_magnitude**2 - mu / radius
    semi_major_axis = -mu / (2 * specific_energy)

    #calculates eccentricity
    e_vector = jnp.cross(velocities, angular_momentum) / mu - positions / radius[:, None]
    eccentricity = jnp.linalg.norm(e_vector, axis=1)

    #calculates right ascension of the ascening node (RAAN)
    node_vector = jnp.cross(z_axis, angular_momentum)
    node_mag = jnp.linalg.norm(node_vector, axis=1)
    raan_raw = jnp.arccos(jnp.clip(node_vector[:, 0] / node_mag, -1.0, 1.0))
    raan = jnp.where(node_vector[:, 1] < 0, 2 * jnp.pi - raan_raw, raan_raw)
    raan_deg = jnp.degrees(raan)

    #calculates argument of periapsis
    n_dot_e = jnp.sum(node_vector * e_vector, axis=1)
    arg_peri_raw = jnp.arccos(jnp.clip(n_dot_e / (node_mag * eccentricity), -1.0, 1.0))
    arg_peri = jnp.where(e_vector[:, 2] < 0, 2 * jnp.pi - arg_peri_raw, arg_peri_raw)
    arg_peri_deg = jnp.degrees(arg_peri)

    #calculates true anomaly
    e_dot_r = jnp.sum(e_vector * positions, axis=1)
    true_anom_raw = jnp.arccos(jnp.clip(e_dot_r / (eccentricity * radius), -1.0, 1.0))
    true_anom = jnp.where(jnp.sum(positions * velocities, axis=1) < 0, 2 * jnp.pi - true_anom_raw, true_anom_raw)
    true_anom_deg = jnp.degrees(true_anom)

    #returns results back to main file
    return (times, positions, radius, velocities, v_magnitude, angular_momentum, angular_magnitude, angular_drift, percent_angular_drift, kinetic_energy, potential_energy, total_energy, energy_drift, percent_energy_drift, 
            inclination_deg, semi_major_axis, eccentricity, raan_deg, arg_peri_deg, true_anom_deg, crash_index)


    




                          
                          
    
