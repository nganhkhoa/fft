include("convolution.jl")

using Images
using TestImages
using FileIO
using Colors
using ImageFeatures
using Formatting

using .Convole

function hough(img, thetaRange=100, rhoRange=100)
  r, c = size(img)
  D = sqrt(r * r + c * c)
  rho_min = -D
  rho_max = D
  L_rho = 2*D
  rho_step = L_rho / rhoRange

  theta_min = -pi/2
  theta_max = pi/2
  L_theta = pi
  theta_step = L_theta / thetaRange

  function dcm_rho(idx)
    rho_min + ((idx) * rho_step) + (rho_step / 2)
  end
  function cdm_rho(rho)
    round(Int32, (rho - rho_min) / rho_step)
  end
  function dcm_theta(idx)
    theta_min + ((idx) * theta_step) + (theta_step / 2)
  end
  function cdm_theta(theta)
    round(Int32, (theta - theta_min) / theta_step)
  end

  accumulator = zeros(Int32, rhoRange, thetaRange)
  #= display(accumulator) =#
  #= println() =#
  for index in CartesianIndices(img)
    x, y = Tuple(index)
    for k in 1:thetaRange
      theta = dcm_theta(k)
      rho = x * cos(theta) + y * sin(theta)
      i = cdm_rho(rho)
      accumulator[i, k] += 1  # simple delta(x,y)
    end
  end

  rho_idx, theta_idx = Tuple(argmax(accumulator))
  rho, theta = dcm_rho(rho_idx), dcm_theta(theta_idx)
  (rho, theta)
end

if abspath(PROGRAM_FILE) == @__FILE__
  #= img = zeros(Bool,10,10) =#
  #= for i in 1:10 =#
  #=   img[2,i] = img[i,2] = img[7,i] = img[i,9] = true =#
  #= end =#
  #= lines = hough_transform_standard(img) =#
  #=  =#
  #= display(img) =#
  #= println() =#
  #= for line in lines =#
  #=   r, t = line =#
  #=   printfmt("x * cos({}π) + y * sin({}π) = {}\n", t/π, t/π, r) =#
  #= end =#

  gaussian_kernel = [
      1 4 7 4 1
      4 16 26 16 4
      7 26 41 26 7
      4 16 26 16 4
      1 4 7 4 1
    ] ./ 273
  laplacian_kernel = [
      -1 -1 -1
      -1  8 -1
      -1 -1 -1
    ]
  sobel_x_kernel = [
      -1 0 1
      -2 0 2
      -1 0 1
  ]
  sobel_y_kernel = rotl90(sobel_x_kernel)

  img = load(ARGS[1])
  img_gray = Gray.(img)
  img_blur = conv(img_gray, gaussian_kernel)
  img_edge_x = conv(img_blur, sobel_x_kernel)
  img_edge_y = conv(img_blur, sobel_y_kernel)
  img_edge = Gray.(round.(Int32, sqrt.((img_edge_x .^ 2) .* (img_edge_y .^ 2))))
  save("hough_edge.jpg", img_edge)
  r, t = hough(img_gray)
  printfmt("x * cos({}π) + y * sin({}π) = {}\n", t/π, t/π, r)
end
 
