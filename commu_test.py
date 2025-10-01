from commu import Communicator

def main():
    comm = Communicator()
    try:
        while True:
            comm.wait_for_signal()
            while (c := input("1=GO, 2=STOP: ")) not in {"1", "2"}: ...
            comm.send_command(crowdedness_status=int(c))
    except KeyboardInterrupt:
        pass
    finally:
        comm.close()

if __name__ == "__main__":
    main()