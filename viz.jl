using Luxor

DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT = 1000, 1000

function not_in_unit(pos)
  return ~(all(0.0 .<= pos .<= 1.0))
end

function visualize(f; width=DEFAULT_IMAGE_WIDTH, height=DEFAULT_IMAGE_HEIGHT)
  @draw f() width height
end

function draw_observation(occupancy_grid_time::Array{Bool,3})
  num_bins = size(occupancy_grid_time)[end]
  T = size(occupancy_grid_time)[1]
  background("white")
  setcolor("black")
  origin()
  
  fig_size = DEFAULT_IMAGE_HEIGHT
  #plot the observed occupancy grid
  cell_size = fig_size ÷ num_bins
  cells = Table(num_bins, num_bins, cell_size, cell_size)
  for t in 1:T
      for (r, c) in [(l[1], l[2]) for l in findall(occupancy_grid_time[t, :, :])]
          Luxor.box(cells[r, c], cell_size, cell_size, :fill)
      end
  end
end

function draw_object_trajectory(tr::Array{<:Real, 2})
  fig_size = DEFAULT_IMAGE_HEIGHT
  path_to_draw = Vector{Point}()
  T = size(tr,1)
  tr = [tr[:,2] tr[:,1]] #needed to reverse 1 and 2 because in luxor first coordinate is colunm
  for t = 1:T
    pos = tr[t,:]
    if not_in_unit(pos)
      continue
    end
    push!(path_to_draw, Point(trunc.(Int, fig_size .* pos)...))
  end
  prettypoly(path_to_draw, action=:stroke)
end

function draw_object_trajectories_single_color(xs::Array{<:Real, 3}; color="green3", overlay=false)
  num_objects = size(xs)[2]
  
  opacity = overlay ? 0.1 : 1.0

  setcolor(sethue(color)..., opacity)
  origin(Point(0, 0))
  
  for o = 1:num_objects
    draw_object_trajectory(xs[:,o,:])
  end
end

function draw_object_trajectories_different_colors(xs::Array{<:Real, 3}; 
                  colors=("brown3", "green3", "darkorange2", "lightslateblue", "blue1"))
  num_objects = size(xs)[2]
  origin(Point(0, 0))
  for o = 1:num_objects
    setcolor(colors[o])
    draw_object_trajectory(xs[:,o,:])
  end
end
