all_industries = ['Industrials', 'Health Care', 'Information Technology', 'Utilities',
 'Financials', 'Materials', 'Consumer Discretionary', 'Real Estate',
 'Communication Services', 'Consumer Staples', 'Energy']

selected_sectors = ["Energy", "Financials", "Industrials", "Information Technology"]
max_foundation_year = 2010
start_date = "2000-01-01"
end_date = "2024-03-01"
test_start_date = "2021-01-01"

tickers_list_file = rf"D:\Trading\sp500_symbols_list.csv"
sp500_stock_file = rf"D:\Trading\raw_data\special\NSDQ.csv"
nsdq_stock_file = rf"D:\Trading\raw_data\special\SP500.csv"
raw_data_dir = rf"D:\Trading\raw_data\tickers"
features_dir = rf"D:\Trading\features"
train_data_dir = rf"D:\Trading\train_data"