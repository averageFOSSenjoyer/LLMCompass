import csv

# Open the CSV file
with open('topo_results_bs1024_ar.csv', mode='r') as infile:
    reader = csv.reader(infile)
    header = next(reader)  # Save the header
    rows = list(reader)  # Read the rows

sorted_rows = sorted(rows, key=lambda row: float(row[0]))

for row in sorted_rows:
    print(row[2])
