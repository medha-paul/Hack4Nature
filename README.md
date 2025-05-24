# Hack4Nature
Team name: Little Angels  
Project Title: "Integrating Dynamic Modeling with Physics Informed Neural Network for Biomass prediction in Waste Water Treatment Plant"

$$
\frac{dS}{dt} = -\frac{1}{Y} \cdot \frac{\mu(t) S}{K_s + S} \cdot X
$$

$$
\frac{dX}{dt} = \frac{\mu(t) S}{K_s + S} \cdot X
$$  
Where:
- `S(t)` is the substrate concentration,  
- `X(t)` is the biomass concentration,  
- `μ(t)` is a time-varying growth rate,  
- `Y` is the yield coefficient,  
- `K_s` is the Monod constant.  
