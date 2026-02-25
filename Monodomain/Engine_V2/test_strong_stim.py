from interactive_simulation import InteractiveSimulation
import numpy as np

sim = InteractiveSimulation(domain_size=80.0, resolution=0.5, 
                            initial_stim_amplitude=200.0, initial_stim_radius=10.0)
sim.add_stimulus(40.0, 40.0)

print(f"Running with 200mV stimulus...")
for step in range(5000):  # 50ms
    I_stim = sim.get_current_stimulus()
    sim.step(sim.dt, I_stim)
    
    if step % 500 == 0:
        V_min_phys = sim.ionic_model.voltage_to_physical(np.array([[np.min(sim.V)]]))[0,0]
        V_max_phys = sim.ionic_model.voltage_to_physical(np.array([[np.max(sim.V)]]))[0,0]
        print(f"t={step*sim.dt:5.1f}ms: V∈[{V_min_phys:7.1f}, {V_max_phys:7.1f}]mV", end='')
        if np.max(sim.V) > 0.8: print(" ✓ AP!", end='')
        if np.min(sim.V) < -0.5: print(" ⚠️ V_min<-0.5!", end='')
        print()

print(f"Final: V_max={np.max(sim.V):.4f}")
