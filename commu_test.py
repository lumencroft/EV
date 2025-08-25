from commu import Communicator

def main():
    comm = Communicator()
    
    try:
        while True:
            comm.wait_for_signal()
            
            user_choice = 0
            while user_choice not in [1, 2]:
                try:
                    raw_input = input("Enter crowdedness status (1 for GO, 2 for STOP): ")
                    user_choice = int(raw_input)
                except ValueError:
                    print("Invalid input. Please enter 1 or 2.")
            
            comm.send_command(crowdedness_status=user_choice)
            
    except KeyboardInterrupt:
        print("\nTest interrupted by user. Closing socket.")
    finally:
        comm.close()

if __name__ == '__main__':
    main()