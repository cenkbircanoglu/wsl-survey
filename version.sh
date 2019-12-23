# Example version script.
# Please choose one version or create your own

grep "version=" setup.py | cut -d "'" -f2
