using FileIO
using Images, ImageView
using Colors
using Formatting
using FFTW

function idealFilter(r, isLow, threshold)
  if isLow
    r .<= threshold
  else
    r .> threshold
  end
end

function gaussianFilter(r, isLow, threshold)
  if isLow
    exp.(-(r.^2) ./ 2*(threshold^2))
  else
    1 .- exp.(-(r.^2) ./ 2*(threshold^2))
  end
end

function butterworthFilter(r, isLow, threshold, n)
  if isLow
    1 ./ (1 .+ (r ./ threshold) .^ (2 * n))
  else
    1 ./ (1 .+ (threshold ./ r) .^ (2 * n))
  end
end

function laplacianFilter(r, isLow, threshold)
  -r.^2
end

function filterFactory(name, shape, isLow, threshold, other)
  n, m = shape
  r = sqrt.([i^2 + j^2 for i=-n/2:n/2-1, j=-m/2:m/2-1])

  if name == "Ideal"
    idealFilter(r, isLow, threshold)
  elseif name == "Gaussian"
    gaussianFilter(r, isLow, threshold)
  elseif name == "Butterworth"
    butterworthFilter(r, isLow, threshold, other)
  elseif name == "Laplacian"
    laplacianFilter(r, isLow, threshold)
  else
    error("Filter not found")
  end
end

if abspath(PROGRAM_FILE) == @__FILE__
  imagePath = ARGS[1]
  filterName = ARGS[2]
  filterLow = ARGS[3] .== "true"
  filterThreshold = parse(UInt32, ARGS[4])
  filterOther = parse(UInt32, ARGS[5])

  img = load(imagePath)
  img_gray = Float32.(Gray.(img))
  filter = filterFactory(filterName, size(img_gray), filterLow, filterThreshold, filterOther)
  #= display(filter) =#
  #= println() =#
  img_fft = fftshift(fft(img_gray))
  out = abs.(real.(ifft(img_fft .* filter)))
  filename = format("filter_{}_{}_{}_{}.png", imagePath, filterName, filterLow, filterThreshold)
  save(filename, out)
end
