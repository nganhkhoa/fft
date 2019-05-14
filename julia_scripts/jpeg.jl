using Images
using FFTW
using Formatting


LUMINANCE_QUANTIZATION_TABLE = [
    16  11  10  16  24   40   51   61   
    12  12  14  19  26   58   60   55   
    14  13  16  24  40   57   69   56   
    14  17  22  29  51   87   80   62   
    18  22  37  56  68   109  103  77   
    24  35  55  64  81   104  113  92   
    49  64  78  87  103  121  120  101  
    72  92  95  98  112  100  103  99   
  ]

ZIGZAG = [
    0   1   5   6  14  15  27  28
    2   4   7  13  16  26  29  42
    3   8  12  17  25  30  41  43
    9  11  18  24  31  40  44  53
    10  19  23  32  39  45  52  54
    20  22  33  38  46  51  55  60
    21  34  37  47  50  56  59  61
    35  36  48  49  57  58  62  63
   ] .+ 1

function to_zigzag(block)
  out_block = zeros(eltype(block), size(ZIGZAG))
  for idx in CartesianIndices(ZIGZAG)
    out_block[ZIGZAG[idx]] = block[idx]
    # display(out_block)
  end
  reshape(out_block, (1,64))
end



function RLC(block)
  codes = Tuple{Int32, Int32, Int32}[]
  leadingZeros = 0
  for idx in CartesianIndices(ZIGZAG)
    if (idx == CartesianIndex(1, 1))
      DC = 0
      push!(codes, (0, DC, block[idx]))
    elseif (block[idx] == 0)
      leadingZeros += 1
    else
      push!(codes, (leadingZeros, log(2, block[idx]), block[idx]))
      leadingZeros = 0
    end
  end
  push!(codes, (0, 0, 0))
  codes
end

if abspath(PROGRAM_FILE) == @__FILE__
  img = load(ARGS[1])
  img_ycbcr = channelview(YCbCr.(img))

  for c in 1:3
    channel = img_ycbcr[c,:,:]
    row, col = size(channel)

    printfmt("{} {}\n", row, col)
    printfmt("{} {}\n", row % 8, col % 8)

    for i in 1:8:row
      for j in 1:8:col
        # printfmt("{} {}\n", i, j)
        block = channel[i:i+7, j:j+7]

        block_shifted = block .- 127
        block_dct = FFTW.dct(block_shifted)

        block_quantization = round.(Int32, block_dct ./ LUMINANCE_QUANTIZATION_TABLE)
        block_zigzag = to_zigzag(block_quantization)

        if i == 1 && j == 9
          display(block)
          println()
          display(block_shifted)
          println()
          display(block_dct)
          println()
          display(block_quantization)
          println()
          display(block_zigzag)
          println()
        end

        # block_zigzag = collect(block_zigzag)
        DC = block_zigzag[1]
        AC = block_zigzag[2:end]
      end
    end
  end
end
