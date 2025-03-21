# Define the directory path
dir="data/formatted"

# Check if the directory exists
if [ -d "$dir" ]; then
    rm -rf "${dir}"
fi

mkdir -p "$dir"
mkdir -p "${dir}/phase_1/"
mkdir -p "${dir}/phase_2/"

# Convert DateTime to timestamp ===============================================
for file in \
    data/raw/phase_1/Training_dataset_1.csv \
    data/raw/phase_1/Training_dataset_2.csv \
    data/raw/phase_1/Training_dataset_3.csv \
    data/raw/phase_1/Test_dataset_1.csv \
    data/raw/phase_1/Test_dataset_2.csv \
    data/raw/phase_1/Test_dataset_3.csv
do
    python preprocess/dt_to_int.py \
        $file \
        data/formatted/phase_1/$(basename $file) \
        --dt_col="DateTime"
done

for file in \
    data/raw/phase_2/FineTune_Train_dataset_1.csv \
    data/raw/phase_2/FineTune_Train_dataset_2.csv \
    data/raw/phase_2/FineTune_Test_dataset_1.csv \
    data/raw/phase_2/FineTune_Test_dataset_2.csv
do
    python preprocess/dt_to_int.py \
        $file \
        data/formatted/phase_2/$(basename $file) \
        --dt_col="DateTime"
done

# Feature engineering =========================================================
for file in \
    data/formatted/phase_1/Training_dataset_1.csv \
    data/formatted/phase_1/Training_dataset_2.csv \
    data/formatted/phase_1/Training_dataset_3.csv \
    data/formatted/phase_1/Test_dataset_1.csv \
    data/formatted/phase_1/Test_dataset_2.csv \
    data/formatted/phase_1/Test_dataset_3.csv
do
    python preprocess/feat_eng.py \
        $file \
        data/formatted/phase_1/$(basename $file)
done

for file in \
    data/formatted/phase_2/FineTune_Train_dataset_1.csv \
    data/formatted/phase_2/FineTune_Train_dataset_2.csv \
    data/formatted/phase_2/FineTune_Test_dataset_1.csv \
    data/formatted/phase_2/FineTune_Test_dataset_2.csv
do
    python preprocess/feat_eng.py \
        $file \
        data/formatted/phase_2/$(basename $file)
done

# Remove NaN rows =============================================================
for file in \
    data/formatted/phase_1/Training_dataset_1.csv \
    data/formatted/phase_1/Training_dataset_2.csv \
    data/formatted/phase_1/Training_dataset_3.csv \
    data/formatted/phase_1/Test_dataset_1.csv \
    data/formatted/phase_1/Test_dataset_2.csv \
    data/formatted/phase_1/Test_dataset_3.csv
do
    python preprocess/clean_data.py \
        $file \
        data/formatted/phase_1/$(basename $file)
done

for file in \
    data/formatted/phase_2/FineTune_Train_dataset_1.csv \
    data/formatted/phase_2/FineTune_Train_dataset_2.csv \
    data/formatted/phase_2/FineTune_Test_dataset_1.csv \
    data/formatted/phase_2/FineTune_Test_dataset_2.csv
do
    python preprocess/clean_data.py \
        $file \
        data/formatted/phase_2/$(basename $file)
done

# Remove constant/NaN cols ====================================================
python preprocess/remove_nan_cols.py \
    "data/formatted/phase_1/Training_dataset_1.csv" \
    "data/formatted/phase_1/Training_dataset_2.csv" \
    "data/formatted/phase_1/Training_dataset_3.csv" \
    "data/formatted/phase_1/Test_dataset_1.csv" \
    "data/formatted/phase_1/Test_dataset_2.csv" \
    "data/formatted/phase_1/Test_dataset_3.csv" \
    "data/formatted/phase_2/FineTune_Train_dataset_1.csv" \
    "data/formatted/phase_2/FineTune_Train_dataset_2.csv" \
    "data/formatted/phase_2/FineTune_Test_dataset_1.csv" \
    "data/formatted/phase_2/FineTune_Test_dataset_2.csv"

# Split features from target ==================================================
for file in \
    data/formatted/phase_1/Training_dataset_1.csv \
    data/formatted/phase_1/Training_dataset_2.csv \
    data/formatted/phase_1/Training_dataset_3.csv
do
    base_name=$(basename "$file" .csv)
    python preprocess/split_xy.py \
        "$file" \
        "data/formatted/phase_1/${base_name}_x.csv" \
        "data/formatted/phase_1/${base_name}_y.csv"
done

for file in \
    data/formatted/phase_2/FineTune_Train_dataset_1.csv \
    data/formatted/phase_2/FineTune_Train_dataset_2.csv
do
    base_name=$(basename "$file" .csv)
    python preprocess/split_xy.py \
        "$file" \
        "data/formatted/phase_2/${base_name}_x.csv" \
        "data/formatted/phase_2/${base_name}_y.csv"
done
