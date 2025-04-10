def wrap_position(x, y, screen_width, screen_height, size):
    """
    Wrap position around screen edges to create a toroidal space.
    """
    if x < -size:
        x = screen_width
    elif x > screen_width:
        x = -size

    if y < -size:
        y = screen_height
    elif y > screen_height:
        y = -size

    return x, y
