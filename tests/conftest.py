import sys

# utils.settings parses the command line when it is imported, which would
# choke on pytest's own arguments.
sys.argv = sys.argv[:1]
