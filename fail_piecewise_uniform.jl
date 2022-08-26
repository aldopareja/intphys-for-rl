using Gen

@gen function model()
  x ~ uniform_continuous(0.0,1.0)
  v ~ uniform_continuous(0.0-x,1.0-x)
end

@gen function fail_piecewise_uniform(prev_trace, offset::Float64, addr::Symbol)
  num_bins::Int = 10
  bin_size = 1/num_bins
  
  idx = [categorical([0.5,0.5])  
           for i = 1:num_bins]
  probs = getindex([1.0,100.0], idx)
  probs ./= sum(probs)

  bounds = collect(0.0:bin_size:1.0)
  anchored_bounds = bounds .- offset
  out = {addr} ~ piecewise_uniform(anchored_bounds, probs)
  if out + offset > bounds[end]
    throw(DomainError(out + offset, "greater"))
  end
end

function inference()
  # [Gen.mh(trace, select(:offset, :out)) for _ in 1:10000]
  traces = [Gen.simulate(model, ()) for i=1:10000]
  for t in traces
    selection = select(:x,:v)
    t, _ = Gen.mh(t,fail_piecewise_uniform, (0.0, :x))
    offset = Gen.get_choice(t,:x).retval
    t, _ = Gen.mh(t,fail_piecewise_uniform, (offset, :v))
  end
end

inference()

function test()
  local v
  for i in 1:1000
    if i == 1
      v = 1.0
    else
      v += 2.0
    end
  end
end

test()