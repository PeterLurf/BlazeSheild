# BlazeSheild
A python project that predict an optimal fireline creation for fighting wild fires. (Topological Mapping, Q-Learning, Convex Hull, PDE simulation)

Simulation Contol Documentation:
Base Igition Probability: How fast fire spreads
Wind Sens: How much wind will push the fire
Slope Sense: How much the slope of the terrain will change fire sprea
Kernal & Gaussian : Size of spreading gaussian kernal and its respective variance
Fireline Start Step: wait how many steps untill fireline starts construction
Fireline Speed: how fast fireline is constructed
Prediction Steps: how far the algo sees into the future. ∝ Acc, ∝ 1/Computational speed
Fireline Thickness: how thick the fireline is. Min rec 2, any lower and fire can leak
Sim Steps: how many iterations should the simulation show

When Simulating, the 3 mandatory elements are the city bounds, wind vector, fire start point.
To initialize the conditions. Start by clicking the elements button and select the points on the map.
After each element's position has been marked, click finish selection. 
Repeat for the other 2. "note. there can be infinite many city zones"

Click Run Sim to display simulation

To train RL model, select the optimal training epochs (rec 50000) and start training.
If model is finetuned and you want to extract architecture, click export current model.
If importing model, be sure to test it.

Have Fun playing around! The finished GIF will be save to local system.
<img width="1439" alt="Screenshot 2025-03-02 at 4 30 25 AM" src="https://github.com/user-attachments/assets/a6437ea4-abb9-4af3-9e7c-bbfc265f2845" />
