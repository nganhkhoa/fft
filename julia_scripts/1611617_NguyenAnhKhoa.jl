using Images
using Formatting
using StatsBase
using FFTW

function genMatrix(what, params...)
  params = map(x->parse(Float32, x), params)
  M = zeros(Float32, (3,3))
  M[3, 3] = 1

  if what == "scale"
    if length(params) < 2
      error("Scale requires two parameter (x, y)")
    end
    x, y = params
    printfmt("Scale with x = {}; y = {}\n", x, y)
    M[1, 1] = x
    M[2, 2] = y
  elseif what == "translate"
    if length(params) < 2
      error("Translate requires two parameter (dx, dy)")
    end
    dx, dy = params
    M[1, 1] = 1
    M[2, 2] = 1
    M[1, 3] = dx
    M[2, 3] = dy
  elseif what == "rotate"
    if length(params) < 1
      error("Rotate requires one parameter (theta, isRadian: true)")
    end
    theta = params[1]
    isRadian = 1
    if length(params) >= 2
      isRadian = params[2]
    end

    if isRadian == 1
      M[1, 1] =  cos(theta)
      M[1, 2] =  -sin(theta)
      M[2, 1] =  sin(theta)
      M[2, 2] =  cos(theta)
    else
      M[1, 1] =  cosd(theta)
      M[1, 2] =  -sind(theta)
      M[2, 1] =  sind(theta)
      M[2, 2] =  cosd(theta)
    end
  elseif what == "shear"
    if length(params) < 4
      error("Rotate requires four parameter (a, b, c, d)")
    end
    a,b,c,d = params
    M[1, 1] = a
    M[1, 2] = b
    M[2, 1] = c
    M[2, 2] = d
  else
    error(printfmt("Matrix of {} is unknown\n", what))
  end
  M
end

function conv(in, kernel)

  row, col = size(in)
  krow, kcol = size(kernel)
  rowPad, colPad = round(Int32, krow/2, RoundDown),round(Int32, kcol/2, RoundDown)

  # feature_maps = zeros(eltype(in), (row + rowPad * 2, col + colPad * 2))
  feature_maps = [
    5 5 0 0 1 2 2 
    5 5 0 0 1 2 2 
    2 2 1 5 1 2 2 
    7 7 1 5 1 2 2 
    7 7 4 5 4 3 3 
    7 7 1 6 1 3 3 
    7 7 1 6 1 3 3
  ]
  out = zeros(eltype(in), size(in))

  colStartIdx = colPad + 1
  colEndIdx = colPad + col
  rowStartIdx = rowPad + 1
  rowEndIdx = rowPad + row

  # setup feature map and kernel
  feature_maps[rowStartIdx:rowEndIdx, colStartIdx:colEndIdx] = in
  kernel = rot180(kernel)

  for i in rowStartIdx:rowEndIdx
    for j in colStartIdx:colEndIdx
      part = feature_maps[i-rowPad:i-rowPad+krow-1, j-colPad:j-colPad+kcol-1]
      v = sum(part .* kernel)
      if (v < 0)
        v = 0
      elseif (v > 1)
        v = 1
      end
      out[i-rowPad,j-colPad] = v
    end
  end
  # TODO: normalize output
  out ./ maximum(out)
end
if abspath(PROGRAM_FILE) == @__FILE__
  I_cau1 = [
    5 0 0 1 2
    2 1 5 1 2
    7 1 5 1 2
    7 4 5 4 3
    7 1 6 1 3
  ]

  sobel_x_kernel = [
      -1 0 1
      -2 0 2
      -1 0 1
  ]
  sobel_y_kernel = rotl90(sobel_x_kernel)

  # I_Gx = imfilter(I_cau1, centered(reflect(sobel_x_kernel)), "reflect")
  # I_Gy = imfilter(I_cau1, centered(reflect(sobel_y_kernel)), "reflect")
  I_Gx = conv(I_cau1, sobel_x_kernel)
  I_Gy = conv(I_cau1, sobel_y_kernel)
  # I_G = round.(Int32, sqrt.((I_Gx .^ 2) .* (I_Gy .^ 2)))

  printfmtln("Vector G(0,0) = ({}, {})", I_Gx[1,1], I_Gy[1,1])
  printfmtln("Vector G(1,1) = ({}, {})", I_Gx[2,2], I_Gy[2,2])
  printfmtln("Vector G(0,3) = ({}, {})", I_Gx[1,4], I_Gy[1,4])

  hist = fit(Histogram, reshape(I_cau1, (length(I_cau1))), nbins=8).weights
  normalized_hist = hist ./ sum(hist)
  cumsum_normalized_hist = cumsum(normalized_hist)

  println("Histogram normalized")
  println(normalized_hist)
  println("Cumsum Histogram normalized")
  println(cumsum_normalized_hist)

  prefered_hist = fit(Histogram, 0:7, nbins=8).weights
  normalized_prefered_hist = prefered_hist ./ sum(prefered_hist)
  cumsum_normalized_prefered_hist = cumsum(normalized_prefered_hist)

  println("Prefered Histogram")
  println(normalized_prefered_hist)
  println("Cumsum Prefered Histogram")
  println(cumsum_normalized_prefered_hist)

  # https://dsp.stackexchange.com/questions/16166/histogram-matching-of-two-images-using-cdf
  out_hist = zeros(8)
  out_color = zeros(8)

  for i in 1:length(out_color)
    for k in 1:length(cumsum_normalized_prefered_hist)
      if cumsum_normalized_prefered_hist[k] - cumsum_normalized_hist[i] < 0
        continue
      end
      out_hist[i] = cumsum_normalized_prefered_hist[k]
      out_color[i] = k
      break
    end
  end

  println("out hist")
  println(out_hist)
  println("out color")
  println([i for i=0:7])
  println(out_color)

  for idx in CartesianIndices(I_cau1)
    newcolor = out_color[I_cau1[idx] + 1]
    I_cau1[idx] = newcolor
  end

  display(I_cau1)
  println()
  hist = fit(Histogram, reshape(I_cau1, (length(I_cau1))), nbins=8).weights
  normalized_hist = hist ./ sum(hist)
  cumsum_normalized_hist = cumsum(normalized_hist)

  println("New Histogram normalized")
  println(normalized_hist)
  println("New Cumsum Histogram normalized")
  println(cumsum_normalized_hist)

  println("==============================")

  fx = [1 3]
  N = length(fx)
  dft_fx = [exp(-2im*pi*u*x / N) for u=0:N-1, x=0:N-1]
  fx_dft = dft_fx * fx'

  println("Ma tran he co dft")
  display(dft_fx)
  println()
  println("Bien doi fft cua fx = [1, 3]")
  println(fx_dft)

  println("==============================")

  function alpha(u, N)
    if u == 0
      sqrt(1/N)
    else
      sqrt(2/N)
    end
  end

  fx = [1 0 1 0] # pad with 0
  N = 4
  dct_fx = [alpha(u, N) * cos((2*x + 1) * u * pi / (2N)) for u=0:N-1, x=0:N - 1]
  fx_dct = dct_fx * fx'

  println("Ma tran he co dct voi N = 4 tren fx")
  display(dct_fx)
  println()
  println("Bien doi dct 4 diem cua fx = [1, 0, 1]")
  println(fx_dct)

  println("==============================")

  M_move_00 = genMatrix("translate", "-3", "-3")
  M_rotate_25 = genMatrix("rotate", "45", "0")
  M_move_33 = genMatrix("translate", "3", "3")

  M = M_move_33 * M_rotate_25 * M_move_00

  println("Ma tran M de xoay hinh xung quanh (3,3)")
  display(M)
  println()

  new_idx = round.(Int32, inv(M) * [2 3 1]')
  printfmtln("Toa do Io(2, 3) co gia tri bang voi I({}, {})", new_idx[1:2]...)

end


# SAMPLE OUTPUT
#=

Vector G(0,0) = (0.0, 0.0)
Vector G(1,1) = (1.0, 1.0)
Vector G(0,3) = (1.0, 0.0)
Histogram
Dict{Int64,Float64} with 8 entries:
  0 => 0.08
  4 => 0.08
  7 => 0.12
  2 => 0.16
  3 => 0.08
  5 => 0.16
  6 => 0.04
  1 => 0.28
==============================
Ma tran he co dft
2×2 Array{Complex{Float64},2}:
 1.0-0.0im   1.0-0.0im        
 1.0-0.0im  -1.0-1.22465e-16im
Bien doi fft cua fx = [1, 3]
2×1 Array{Complex{Float64},2}:
  4.0 + 0.0im                   
 -2.0 - 3.6739403974420594e-16im
==============================
Ma tran he co dct voi N = 4 tren fx
4×3 Array{Float64,2}:
 0.5        0.5        0.5     
 0.653281   0.270598  -0.270598
 0.5       -0.5       -0.5     
 0.270598  -0.653281   0.653281
Bien doi dct 4 diem cua fx = [1, 0, 1]
4×1 Array{Float64,2}:
 1.0               
 0.3826834323650898
 0.0               
 0.9238795325112867
==============================
Ma tran M de xoay hinh xung quanh (3,3)
3×3 Array{Float32,2}:
 0.707107  -0.707107   3.0    
 0.707107   0.707107  -1.24264
 0.0        0.0        1.0    
Toa do Io(2, 3) co gia tri bang voi I(2, 4)

=#
