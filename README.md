# Hack4Nature
$\textbf{Team name:}$ Little Angels  

$\textbf{Project Title:}$ "Integrating Dynamic Modeling with Physics Informed Neural Network for Biomass prediction in Waste Water Treatment Plant"  

$\textbf{Background:}$ Biomass formation and its proper management is critical for the output of the waste water treatment plant. The right biomass concentration is the key as excess concentration can lead to generate more waste and thereby increasing the pollution as well as opearational cost of the plant.  

$\textbf{Novelty:}$ Here, we tried to modify the classical MONOD kinetics for biomass concentration by introducing $\mu(t)$. Then we have applied Physics Informed Neural network (PINN) algorithm to estimate the biomass concentration at any point of time. This algorithm is efficient in capturing solution even with small amount of data.

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
