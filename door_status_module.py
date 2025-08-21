call_count = 0

def get_door_status(frame):
    global call_count
    call_count += 1

    if call_count <= 5:
        door_status = 2
    else:
        door_status = 1
        call_count = 0

    return door_status