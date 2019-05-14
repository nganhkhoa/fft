using FileIO
using Images, ImageView
using Formatting

function roundAround(x, low, high)
  if x < low
    low
  elseif x > high
    high
  else
    round(Int32, x)
  end
end

function forwardTransform(M, P)
  n, m = size(P)
  out = zeros(eltype(P), n, m)
  for index in CartesianIndices(P)
    x, y = Tuple(index)
    newindex = M * [x; y; 1]
    x1 = roundAround(newindex[1], 1, n)
    y1 = roundAround(newindex[2], 1, m)
    out[x1, y1] = P[x, y]
  end
  out
end

function backwardTransform(algo, M, P)
  M = inv(M)
  n, m = size(P)
  out = zeros(eltype(P), n, m)
  for index in CartesianIndices(out)
    x, y = Tuple(index)
    sourceIndex = M * [x; y; 1]
    sourceIndex ./= sourceIndex[3]
    x1 = round(Int32, sourceIndex[1])
    y1 = round(Int32, sourceIndex[2])
    #= if ((x == 1 && y == 1) || =#
    #=     (x == 1 && y == m) || =#
    #=     (x == n && y == 1) || =#
    #=     (x == n && y == m)) =#
    #=   println("AAAAAAAAAAA") =#
    #=   display([x; y; 1]) =#
    #=   println() =#
    #=   display(sourceIndex) =#
    #=   println() =#
    #= end =#
    if (x1 < 1 || y1 < 1 || x1 > n || y1 > m)
      continue
    end
    if (sourceIndex[1] > n || sourceIndex[2] > m)
      continue
    end
    out[x, y] = P[x1, y1]
  end
  out
end

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

function projectiveTransform(picture, points)
  if length(points) != 4
    error("Not correct number of point")
  end

  A = zeros(Int64, 2*length(points), 8)
  B = zeros(Int64, 2*length(points), 1)
  for i in 1:length(points)
    x, y, x1, y1 = points[i]
    A[2*i - 1,:] = [x y 1 0 0 0 -x*x1 -x1*y]
    A[2*i,:] = [0 0 0 x y 1 -x*y1 -y*y1]
    B[2*i - 1] = x1
    B[2*i] = y1
  end
  H = vcat(A\B, 1)
  M = reshape(H, (3,3))'

  println("Matrix M = reshape([A\\B, 1], (3,3))")
  display(M)
  println()

  backwardTransform("zero", M, picture)
end

if abspath(PROGRAM_FILE) == @__FILE__
  what = ARGS[1]
  img = load(ARGS[2])

  if what == "transform"
    # M = [[cosd(25) -sind(25) 0]; [sind(25) cosd(25) 0]; [0 0 1]]
    # M = genMatrix("rotate", 25, false)

    M = genMatrix(ARGS[3], ARGS[4:end]...)

    forward = forwardTransform(M, img)
    save("forward.png", forward)

    backward = backwardTransform("zero", M, img)
    save("backward.png", backward)
  elseif what == "multitransform"
  elseif what == "projective"
    row, col = size(img)
    out = projectiveTransform(img,
      [
        # [x y x1 y1],
        [55 187 1 1],
        [1276 619 1295 1],
        [1447 1761 1295 1635],
        [166 1819 1 1635]
      ]
    )
    save("projective.png", out)
  else
    println("Do what? transform/projective")
  end

end
