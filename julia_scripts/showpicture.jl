module Showpicture

using Images, ImageView, Gtk.ShortNames

export showpicture

function showpicture(img)

  guidict = imshow(img);
  #If we are not in a REPL
  if (!isinteractive())
    c = Condition()
    # Get the window
    win = guidict["gui"]["window"]
    # Notify the condition object when the window closes
    signal_connect(win, :destroy) do widget
        notify(c)
    end
    # Wait for the notification before proceeding ...
    wait(c)
  end
end

end
