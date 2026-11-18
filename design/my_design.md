The paper used a nominal LQR and a low level controller (for torque control).
And MPC is used to as a corrective input to correct for wind disturbance which
leaves less room for analysis of Koopman or the MPC design

**My Idea** 
- Replace the LQR with MPC completely. MPC controls the entire plant and
generate the high level control command $u_h = [\omega_x, \omega_y \omega_z, f]$
- The MPC used a model which is identified by Koopman using data generated from a 
flight that is carried out in wind (in simulation)
- Create two models from Koopman i) Linear Model ii) Bilinear mode
- Design a MPC controller with both the models for reference trajectory tracking
- Design proper terminal cost and terminal constraints for the MPC and establish 
recursive stability
- Compare the paper's K-MPC (model 1) vs Full MPC linear model vs Full MPC bilinear model
