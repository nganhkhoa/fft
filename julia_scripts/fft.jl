include("showpicture.jl")

using FFTW
using Images
using ImageView
using Plots

using .Showpicture

function dft(f)
  if ndims(f) == 1
    M = length(f) 
    [sum([x * exp(-2im*pi*u * (idx - 1) / M ) for (idx, x) in enumerate(f)]) for u=0:M-1]
  elseif ndims(f) == 2
    M, N = size(f)
    [sum([f[x+1, y+1] * exp(-2im*pi*(u*x/M + v*y/M)) for x=0:M-1, y=0:N-1]) for u=0:M-1, v=0:N-1]
  else
    error("Wrong dimension")
  end

  # mesh = [i * j for i=0:N-1, j=0:N-1]
  # W = exp.(-2im * pi * mesh / N)
  # W * f

end

function fft1d(array)
  #= https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm
  #
  # Xk = Ek + e^(-2pi*i/N * k) * Ok
  #
  # Ek = Event DFT
  # Ok = Odd DFT
  #
  =#
  N = size(array)[1]
  if N == 1
    return array
  elseif N % 2 != 0
    error("Wrong dimension")
  end

  # people count from zero while julia counts from 1
  # either way, it should be the same name
  even = fft1d(array[1:2:end])
  odd = fft1d(array[2:2:end])
  w = exp.(-2im * pi / N * [i for i=0:N/2-1])
  half1 = even .+ (odd .* w)
  half2 = even .- (odd .* w)
  vcat(half1,half2)
end

function fft2d(matrix)
  if ndims(matrix) != 2
    error("I dont handle other than 2 dimension matrix")
  end
  matrix_fft1 = hcat(fft1d.(eachrow(matrix))...)
  hcat(fft1d.(eachrow(matrix_fft1))...)
end

if abspath(PROGRAM_FILE) == @__FILE__
  if (length(ARGS) < 1)
    println("fft library")
    display(fft(1:8))
    println()
    println("fft own implemenation")
    display(fft1d(1:8))
    println()
    println("DFT1D")
    display(dft(1:8))
    println()

    println("====================")

    println("fft2d library")
    display(fft([[1:8...] [1:8...]]'))
    println()
    println("fft2d own implemenation")
    display(fft2d([[1:8...] [1:8...]]'))
    println()
    println("DFT2D")
    display(dft([[1:8...] [1:8...]]'))
    println()
  else
    img = load(ARGS[1])
    img_gray = Float32.(Gray.(img))

    # img_fft = fft2d(Gray.(img))
    img_fft = fftshift(fft(img_gray))
    normalize = log.(abs.(img_fft)) .+ 1
    showpicture(normalize)
  end
end
