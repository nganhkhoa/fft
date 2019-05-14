using FFTW

function dct(f)
  function a(u, N)
    if u == 0
      sqrt(1/N)
    else
      sqrt(2/N)
    end
  end

  if ndims(f) == 1
    N = length(f)
    [a(u, N) * sum([x * cos(pi * u * (2i - 1)/ (2N)) for (i, x) in enumerate(f)]) for u=0:N - 1]
  elseif ndims(f) == 2
    N, M = size(f)
    [a(u, N) * a(v, M) * sum([f[x+1, y+1] * cos(pi * u * (2x + 1)/ (2N)) * cos(pi * v * (2y + 1) / (2M)) for x=0:N-1, y=0:M-1]) for u=0:N - 1, v=0:M-1]
  else
    error("What the hell man?")
  end
end

if abspath(PROGRAM_FILE) == @__FILE__
  display(dct(1:8))
  println()
  display(FFTW.dct(1:8))
  println()

  display(dct([(i-1)*8 + j for i=1:8, j=1:8]))
  println()
  display(FFTW.dct([(i-1)*8 + j for i=1:8,j=1:8]))
  println()
end
