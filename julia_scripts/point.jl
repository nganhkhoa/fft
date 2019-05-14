using Images
using FileIO
using Colors

function transform(img, what, params...)
  if length(params) < 2
    error("Transform requires two parameter (a, b)")
  end
  a, b = params
  out = zeros(eltype(img), size(img))
  if what == "linear"
    out = a .* img .+ b
  elseif what == "log"
    out = a .* log.(1 .+ img) .+ b
  elseif what == "exp"
    out = a .* exp.(img) .+ b
  elseif what == "pow"
    if length(params) < 3
      error("Power transform requires three parameter (a, b, gamma)")
    end
    gamma = params[3]
    out = a .* (img .^ gamma) .+ b
  else
    error("What the hell man?")
  end
  # TODO: Scale to 255
  out
end

if abspath(PROGRAM_FILE) == @__FILE__
  what = ARGS[1]
  img = Gray.(load(ARGS[2]))
  out = transform(img, what, ARGS[3:end]...)
  save("point.png", out)
end
