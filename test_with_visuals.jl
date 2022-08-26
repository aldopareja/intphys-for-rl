# need to include most of `voxel_based_smc.jl` (function definitions and using clauses) so that the code here runs
using Seaborn
"""
visualizing velocity proposal

to do this I'll sample velocities with the piecewise_2d using the 1st postition as anchor, and then plotting the implied second position. 
"""

function test_velocity_proposal(trace)
  high_score = 10000.0
  xs, occupancy_grid_time = Gen.get_retval(trace)
  samples = [vel_t_proposal(trace, 2, 1, occupancy_grid_time, high_score)
             for i=1:1000]
  x2s = [xs[2,1,:]] .+ samples
  x2s = reduce(vcat, transpose.(x2s))
  x1s = reduce(vcat, [xs[2,1,:]' for _=1:1000])
  #stack to get trajectories
  trs = permutedims(cat(x1s,x2s;dims=3), [3,1,2])
  visualize() do
    draw_observation(occupancy_grid_time)
    draw_object_trajectories_single_color(trs; color="red", overlay=true)
  end
  displot(x=trs[2,:,2], y=1 .- trs[2,:,1])
  display(gcf())
end


