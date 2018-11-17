def compare_squares(current_value, next_value, thw=60, thh=60):
    # compare distances for width
    joined = False
    if abs(current_value[2]-next_value[0]) < thw and abs(current_value[1]-next_value[1]) < thh: 
        
        x_peque = current_value[0]
        y_peque = current_value[1]
        x_gran = next_value[2]
        y_gran = next_value[3]
            
        current_value = [x_peque, y_peque, x_gran, y_gran]
        
        joined = True
        
    return current_value, joined
    
    