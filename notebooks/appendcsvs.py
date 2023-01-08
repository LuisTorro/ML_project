filenames = os.listdir(PATH)
csv_list = []
for filename in filenames:
    csv = PATH + '\\' +filename
    csv_list.append(csv)

df = pd.concat(map(pd.read_csv, csv_list))