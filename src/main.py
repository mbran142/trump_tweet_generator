from clean import clean_data
from train import build_and_train_model
import sys
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--clean', action="store_true", help='Only clean a raw .json file. Must include rawfile and parsedfile args.')
    parser.add_argument('-r', '--rawfile', help='Raw tweet json file name')
    parser.add_argument('-p', '--parsedfile', help='Parsed tweet json file name')
    parser.add_argument('-n', '--name', help='Name of model')
    parser.add_argument('-t', '--trained', action="store_true", help='Load trained model') #TODO: figure out what kind of file this requires (check 151B PA4)

    args = parser.parse_args()

    # generate tweets using trained model
    if args.trained:

        print('TODO: use trained model here')
        # HANDLE TRAINED MODEL LOGIC HERE:
        # crash program if -c, -r, -p are provided or if -n is not provided
        # otherwise, load the model using the provided filename and begin the tweet-generating user input

    else: 

        # only clean a file
        if args.clean:
            
            if args.rawfile is None or args.parsedfile is None:
                print('Usage Error: -c argument requires both --rawfile and --parsedfile arguments.')
                exit(1)

            if not clean_data(args.rawfile, args.parsedfile):
                exit(1)

        # train model
        else:

            if (args.rawfile is not None and args.parsedfile is not None) or (args.rawfile is None and args.parsedfile is None):
                print('Usage Error: Model training requires either --rawfile or --parsedfile but not both')
                exit(1)

            else:
                
                # rawfile provided but not parsed file
                if args.rawfile is not None:

                    # get filename without potention extension
                    dot_indices = [pos for pos, char in enumerate(args.rawfile) if char == '.']
                    last_dot_index = len(args.rawfile) if len(dot_indices) == 0 else dot_indices[-1]
                    parsed_filename = args.rawfile[0 : last_dot_index] + '_PARSED.json'

                    # save parsed data into now file
                    print('Cleaning raw data. Saving into ' + parsed_filename)
                    if not clean_data(args.rawfile, parsed_filename):
                        exit(1)

                else:
                    parsed_filename = args.parsedfile

                # determine settings file name
                if args.name is None:
                    config_file = 'default.json'
                else:
                    config_file = args.name
                    if '.json' not in config_file:
                        config_file += '.json'

                # train model on parsed file
                print('Training model using config file ' + config_file + ' on parsed data file ' + parsed_filename + '...')

                # TODO: check if all files exists
                # config file
                # parsed filename
                # if either don't exist, report error and exist

                build_and_train_model(config_file, parsed_filename)