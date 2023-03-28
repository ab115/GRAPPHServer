import yahooquery as yfinq
from flask import Flask, jsonify, make_response
import csv
from flask import request
import json
import pandas as pd
import numpy as np
import sys  
sys.path.append('./src')
from edge import *
from edge_risk_kit import *
import edge_risk_kit as erk
from tqdm.notebook import tqdm
import yahooquery as yf
import jsons

from sklearn import preprocessing
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
#from climada.entity import ImpactFunc

app = Flask(__name__)
#cors = CORS(app, resources={r"/*": {"origins": "*"}})

DATA_FOLDER = 'C:\\Pinaki\\Work\\Code\\ESGServer\\.venv\\'

class GraphData:
        gDataBM = []
        gDataPF =[]
        gLabels = []
  
        def __init__(self, pLabels, pDataBM, pDataPF):
                self.gLabels = pLabels
                self.gDataBM = pDataBM
                self.gDataPF = pDataPF

@app.get('/get_esg_scores')
def get_esg_scores():
        csvfile = open('C:\Pinaki\Work\Code\ESGServer\.venv\esg_scores.csv', 'r')
        jsonfile = open('esg_scores.json', 'w')

        fieldnames = ("symbol", "socialScore", "governanceScore", "environmentScore", "totalEsg", "esgPerformance", "percentile", "peerGroup", "highestControversy")
        reader = csv.DictReader(csvfile, fieldnames)
        
        rowcount = 0
        jsonfile.write('[')
        for row in reader:
                if rowcount == 0:
                        rowcount = rowcount + 1
                        continue
                else:
                        json.dump(row, jsonfile, sort_keys=True, indent=4, separators=(',', ':'))                        
                        jsonfile.write(',')
                        jsonfile.write('\n')

        csvfile.close()
        jsonfile.write("{} ]")
        jsonfile.close()
        jsonfile = open('esg_scores.json')

        # returns JSON object as
        # a dictionary
        data = json.load(jsonfile)
              
        response = make_response(data)
        response.headers['Access-Control-Allow-Origin'] = '*' 
        response.content_type = 'application/json'
        return response, 200

@app.get('/get_portfolio')
def get_portfolio():
        csvfile = open('C:\Pinaki\Work\Code\ESGServer\.venv\Portfolio.csv', 'r')
        jsonfile = open('Portfolio.json', 'w')

        fieldnames = ('Symbol', 'Name', 'Sector', 'CurrentPrice', 'ClosedPrice', 'MarketCap', 'Country', 'Weightage', 'ptotalEsg')
        reader = csv.DictReader(csvfile, fieldnames)
        
        rowcount = 0
        jsonfile.write('[')
        for row in reader:
                if rowcount == 0:
                        rowcount = rowcount + 1
                        continue
                else:
                        json.dump(row, jsonfile, sort_keys=True, indent=4, separators=(',', ':'))                        
                        jsonfile.write(',')
                        jsonfile.write('\n')

        csvfile.close()
        jsonfile.write("{} ]")
        jsonfile.close()
        jsonfile = open('Portfolio.json')

        # returns JSON object as
        # a dictionary
        data = json.load(jsonfile)
              
        response = make_response(data)
        response.headers['Access-Control-Allow-Origin'] = '*' 
        response.content_type = 'application/json'
        return response, 200

@app.route('/fetch_esg_scores', methods=['GET'])
def fetch_esg_scores():
        snp = pd.read_csv(DATA_FOLDER + 'Portfolio.csv', encoding='unicode-escape')
        snp.set_index('Symbol', inplace=True)
        snp.head()
        esg_data = pd.DataFrame([])
        sticker = request.args.get("pticker")
        if sticker != None and sticker != '':
                field_names = ("Symbol", "socialScore", "governanceScore", "environmentScore", "totalEsg", "esgPerformance","percentile", "peerGroup", "highestControversy")
                df_esg_existing = pd.read_csv(DATA_FOLDER + 'esg_scores.csv')
                temp = yfinq.Ticker(sticker).esg_scores
                tempdf = pd.DataFrame.from_dict(temp).T
                tempdf['Symbol'] = str(sticker)
                esg_data = pd.concat([esg_data, tempdf])
                str_esg = {'Symbol':(esg_data.loc[str(sticker)]['Symbol']), 'socialScore':(str(esg_data.loc[str(sticker)]['socialScore'])),'governanceScore':(str(esg_data.loc[str(sticker)]['governanceScore'])) , 'totalEsg':(str(esg_data.loc[str(sticker)]['totalEsg'])) ,'esgPerformance':(str(esg_data.loc[str(sticker)]['esgPerformance'])),'percentile':(str(esg_data.loc[str(sticker)]['percentile'])) ,'peerGroup':(str(esg_data.loc[str(sticker)]['peerGroup'])) ,'highestControversy':(str(esg_data.loc[str(sticker)]['highestControversy']))}
                esg_row = [(esg_data.loc[str(sticker)]['Symbol']),esg_data.loc[str(sticker)]['socialScore'], esg_data.loc[str(sticker)]['governanceScore'], esg_data.loc[str(sticker)]['totalEsg'], esg_data.loc[str(sticker)]['esgPerformance'], esg_data.loc[str(sticker)]['percentile'], esg_data.loc[str(sticker)]['peerGroup'], esg_data.loc[str(sticker)]['highestControversy']]
                with open(DATA_FOLDER + 'esg_scores.csv', mode='a', newline='') as file:
                        writer = csv.DictWriter(file, fieldnames=field_names)
                        writer.writerow(str_esg)
                        file.close()
                return jsonify("Successfully fetched ticker ESG"), 200
        for ticker in tqdm(snp.index):
                try:
                        temp = yfinq.Ticker(ticker).esg_scores
                        tempdf = pd.DataFrame.from_dict(temp).T
                        tempdf['Symbol'] = str(ticker)
                        esg_data = pd.concat([esg_data, tempdf])
                except Exception as e:
                        print(e)
                        continue
        esg_data.set_index('Symbol', inplace=True)
        required_cols = ['socialScore', 'governanceScore', 'environmentScore', 'totalEsg',
                                 'esgPerformance', 'percentile', 'peerGroup', 'highestControversy']
        esg_data.columns.name = ''
        esg_data = esg_data[required_cols]
        esg_data = esg_data.apply(pd.to_numeric, errors='ignore')
        esg_data.shape
        esg_data.sort_values('totalEsg', ascending=False).head()
        esg_data.to_csv(DATA_FOLDER + 'esg_scores.csv')

        result = ""
        with open(DATA_FOLDER + 'esg_scores.csv', "r+") as file:
                for line in file:
                        if not line.isspace():
                                result += line
                file.seek(0)
                file.write(result)

        return jsonify("Successfully fetched esg scores"), 200

@app.route('/get_security_data', methods=['GET'])
def get_security_data():
        snp = pd.read_csv(DATA_FOLDER + 'Portfolio.csv', encoding='unicode-escape')
        snp.set_index('Symbol', inplace=True)
        snp.head()
        tickers = snp.index.to_list()
        security_data = pd.DataFrame([])
        sticker = request.args.get('pticker')
        #what is this if doing
        if sticker != None and sticker != '':
                request.form.get('pticker')
                tickers = [str(request.form.get('pticker'))]

        for ticker in tqdm(tickers):
                try:
                        info = yfinq.Ticker(ticker).asset_profile
                        summary = yfinq.Ticker(ticker).summary_detail
                        infodf = pd.DataFrame.from_dict(info).T
                        summarydf = pd.DataFrame.from_dict(summary).T
                        infodf['symbol'] = ticker
                        summarydf['symbol'] = ticker
                        stockdf = pd.merge(infodf, summarydf, on="symbol")
                        stockdf['symbol'] = ticker
                        security_data = pd.concat([security_data, stockdf])
                except Exception as e:
                        print(e)
                        continue

        security_data.shape
        security_data.set_index('symbol', inplace=True)

        if request.args.get('pticker') != None and request.args.get('pticker') != '':
                secdf = pd.read_csv(DATA_FOLDER + 'security_data.csv', encoding='unicode-escape')
                secdf.set_index('symbol', inplace=True)
                secdf = pd.concat([security_data,secdf])
                secdf.to_csv(DATA_FOLDER + 'security_data.csv')
                return jsonify("Added security data"), 200

        security_data.to_csv(DATA_FOLDER + 'security_data.csv')
        return jsonify("Successfully fetched security scores"), 200

@app.route('/get_security_prices', methods=['GET'])
def get_security_prices():
        start_date = '2013-01-01'
        end_date = '2023-03-23'
        snp = pd.read_csv(DATA_FOLDER + 'Portfolio_workshop_draft.csv', encoding='unicode-escape')
        snp.set_index('Symbol', inplace=True)
        tickers = snp.index.to_list()
        data = yf.download(tickers, start=start_date, end=end_date)
        prices = data['Adj Close'][tickers]
        px = prices.loc['2013':].dropna(axis=1, how='all')
        px.to_csv(DATA_FOLDER + 'prices.csv')

        return jsonify("Success building prices"), 200

def do_analytics():        
        snp = pd.read_csv(DATA_FOLDER + 'snp500_constituents.csv')
        snp.set_index('Symbol', inplace=True)
        px = pd.read_csv(DATA_FOLDER + 'prices.csv')
        px.Date = pd.to_datetime(px.Date)
        px.set_index('Date', inplace=True)
        px = px[px.columns[px.count() == px.count().max()]]
        esg_data = pd.read_csv(DATA_FOLDER + 'esg_scores.csv')
        security_data = pd.read_csv(DATA_FOLDER + 'security_data.csv')
        esg_data.set_index('symbol', inplace=True)
        security_data.set_index('symbol', inplace=True)
        rets_monthly, cov_monthly = calcRetsCov(px, 'M')
        rets_period = rets_monthly
        
        PERIODS_PER_YEAR = 12
        RISK_FREE_RATE = 0.013
        risk_data = erk.summary_stats(rets_period, riskfree_rate=RISK_FREE_RATE, periods_per_year=PERIODS_PER_YEAR).sort_values('Sharpe Ratio', ascending=False)
        full_data = risk_data.join(esg_data).join(security_data['marketCap']).join(snp)
        full_data = full_data[~full_data.totalEsg.isnull()]
        full_data['mktcap_grp'] = pd.cut(full_data.percentile, 3, labels=["Small", "Medium", "Large"])
        largePeerGroup = esg_data.peerGroup.value_counts().index[0:20].to_list()
        full_data['peerGroup2'] = full_data.peerGroup.apply(lambda x: x if x in largePeerGroup else 'Others')
        full_data['esg_soc_grp'] = pd.cut(full_data.socialScore, 5,
                                          labels=["Severe Risk", "High Risk", "Medium Risk", "Low Risk", "No Risk"])
        full_data['esg_env_grp'] = pd.cut(full_data.environmentScore, 5,
                                          labels=["Severe Risk", "High Risk", "Medium Risk", "Low Risk", "No Risk"])
        full_data['esg_gov_grp'] = pd.cut(full_data.governanceScore, 5,
                                          labels=["Severe Risk", "High Risk", "Medium Risk", "Low Risk", "No Risk"])
        full_data['esg_tot_grp'] = pd.cut(full_data.totalEsg, 5,
                                          labels=["Severe Risk", "High Risk", "Medium Risk", "Low Risk", "No Risk"])        
        full_data.shape
        return full_data

def calcRetsCov(px, freq):
        px_freq = px.resample(freq).fillna('ffill')
        px_freq.index = px_freq.index.to_period(freq)
        rets = px_freq.pct_change().dropna(axis=1, how='all').dropna()
        cov = rets.cov()
        return rets, cov

@app.get('/do_panalysis')
def do_panalysis():
        full_data = do_analytics()

        score_list = ['socialScore', 'governanceScore', 'environmentScore', 'totalEsg']
        num_of_stocks = 30

        snp = pd.read_csv(DATA_FOLDER + 'snp500_constituents.csv')
        snp.set_index('Symbol', inplace=True)
        px = pd.read_csv(DATA_FOLDER + 'prices.csv')
        px.Date = pd.to_datetime(px.Date)
        px.set_index('Date', inplace=True)
        px = px[px.columns[px.count() == px.count().max()]]

        rets_monthly, cov_monthly = calcRetsCov(px, 'M')

        er_bmk = erk.annualize_rets(rets_monthly, 12)
        cov_bmk = rets_monthly.cov()

        return_bmk = (ew(er_bmk) * rets_monthly).sum(axis=1)
        wealth_bmk = erk.drawdown(return_bmk).Wealth

        return_port = {}
        wealth_port = {}

        return_port['bmk'] = return_bmk
        wealth_port['bmk'] = wealth_bmk

        for score in score_list:
                return_ = {}
                wealth_ = {}

                stock_selected = full_data.sort_values(score, ascending=False).head(num_of_stocks).index

                ## only need the expected return to generate the equal weights... other than that is redundant for now
                er_port = erk.annualize_rets(rets_monthly[stock_selected], 12)
                cov_port = rets_monthly[stock_selected].cov()

                return_ = (ew(er_port) * rets_monthly[stock_selected]).sum(axis=1)
                wealth_ = erk.drawdown(return_).Wealth

                return_port[score] = return_
                wealth_port[score] = wealth_

        return_port = pd.DataFrame(return_port)
        wealth_port = pd.DataFrame(wealth_port)
        
        df = pd.DataFrame()
        df['totalEsg'] = return_port.get("totalEsg")
        df['bmk'] = return_port.get("bmk")
        df['Date1'] = return_port.index.values
        df['labels'] = df['Date1'].dt.month.astype(str) + "-" + df['Date1'].dt.year.astype(str)
        
        data = GraphData(df.get('labels'), df.get("bmk"), df.get("totalEsg"))
        
        response = make_response(jsons.dumps(data))
        response.headers['Access-Control-Allow-Origin'] = '*' 
        response.content_type = 'application/json'
        return response, 200
 
@app.get('/portfolioVsBenchmark')
def portfolioVsBenchmark():
        full_data = do_analytics()

        score_list = ['socialScore', 'governanceScore', 'environmentScore', 'totalEsg']
        num_of_stocks = 30

        snp = pd.read_csv(DATA_FOLDER + 'snp500_constituents.csv')
        snp.set_index('Symbol', inplace=True)
        px = pd.read_csv(DATA_FOLDER + 'prices.csv')
        px.Date = pd.to_datetime(px.Date)
        px.set_index('Date', inplace=True)
        px = px[px.columns[px.count() == px.count().max()]]

        snppx = pd.read_csv(DATA_FOLDER + 'snp500_HistData.csv')
        rets_monthly, cov_monthly = calcRetsCov(px, 'M')

        er_bmk = erk.annualize_rets(rets_monthly, 12)
        cov_bmk = rets_monthly.cov()

        return_bmk = (ew(er_bmk) * rets_monthly).sum(axis=1)
        wealth_bmk = erk.drawdown(return_bmk).Wealth

        return_port = {}
        wealth_port = {}

        return_port['bmk'] = return_bmk
        wealth_port['bmk'] = wealth_bmk

        for score in score_list:
                return_ = {}
                wealth_ = {}

                stock_selected = full_data.sort_values(score, ascending=False).head(num_of_stocks).index

                ## only need the expected return to generate the equal weights... other than that is redundant for now
                er_port = erk.annualize_rets(rets_monthly[stock_selected], 12)
                cov_port = rets_monthly[stock_selected].cov()

                return_ = (ew(er_port) * rets_monthly[stock_selected]).sum(axis=1)
                wealth_ = erk.drawdown(return_).Wealth

                return_port[score] = return_
                wealth_port[score] = wealth_

        return_port = pd.DataFrame(return_port)
        wealth_port = pd.DataFrame(wealth_port)
        
        
        df = pd.DataFrame()
        df['totalEsg'] = snppx.get("Open")
        df['bmk'] = snppx.get("Close")
        df['labels'] = snppx.get("Date")
        data = GraphData(df.get('labels'), df.get("bmk"), df.get("totalEsg"))
        
        response = make_response(jsons.dumps(data))
        response.headers['Access-Control-Allow-Origin'] = '*' 
        response.content_type = 'application/json'
        return response, 200


@app.get('/add_ticker')
def add_ticker():
        ticker = request.args.get('pticker')
        if ticker  != None and ticker  != '':
                sticker = yfinq.Ticker(ticker)
                data_dict = sticker.asset_profile
                sector = data_dict[ticker]['industry']
                longName = data_dict[ticker]['longBusinessSummary']
                lArray = longName.split(',')
                cls = sticker.price[ticker]['regularMarketPreviousClose']
                curr = sticker.price[ticker]['regularMarketPrice']
                marketCap = sticker.price[ticker]['marketCap']
                country = data_dict[ticker]['country']
                finalArray = [ticker, lArray[0], sector]
                field_names = ['Symbol', 'Name', 'Sector', 'CurrentPrice', 'ClosedPrice', 'MarketCap', 'Country']
                dict = {"Symbol": ticker, "Name": lArray[0], "Sector": sector,
                        "CurrentPrice": curr, "ClosedPrice": cls, "MarketCap": marketCap, "Country": country}
                with open (DATA_FOLDER + 'Portfolio.csv','a',newline='') as csv_file:
                        dict_object = csv.DictWriter(csv_file, fieldnames=field_names)
                        dict_object.writerow(dict)
                        csv_file.close()
        else:
                with open(DATA_FOLDER + 'portfolio_all_values.txt') as file:
                        for item in file:
                                try:
                                        ticker = item.strip()
                                        sticker = yfinq.Ticker(ticker)
                                        data_dict = sticker.asset_profile
                                        sector = data_dict[ticker]['industry']
                                        longName = data_dict[ticker]['longBusinessSummary']
                                        lArray = longName.split(',')
                                        cls = sticker.price[ticker]['regularMarketPreviousClose']
                                        curr = sticker.price[ticker]['regularMarketPrice']
                                        marketCap = sticker.price[ticker]['marketCap']
                                        country = data_dict[ticker]['country']
                                        finalArray = [ticker, lArray[0], sector]
                                        field_names = ['Symbol', 'Name', 'Sector', 'CurrentPrice', 'ClosedPrice',
                                                       'MarketCap', 'Country']
                                        dict = {"Symbol": ticker, "Name": lArray[0], "Sector": sector,
                                                "CurrentPrice": curr, "ClosedPrice": cls, "MarketCap": marketCap, "Country": country}
                                        with open('Portfolio.csv', 'a', newline='') as csv_file:
                                                dict_object = csv.DictWriter(csv_file, fieldnames=field_names)
                                                dict_object.writerow(dict)
                                                csv_file.close()
                                except Exception as e:
                                        print(e)
                                        continue

        if ticker  != None and ticker  != '':
                add_esg_to_portfolio()
                add_weightage_to_stocks_in_portfolio()
        
        result = ""
        with open(DATA_FOLDER + "Portfolio.csv", "r+") as file:
                for line in file:
                        if not line.isspace():
                                result += line
                file.seek(0)
                file.write(result)
        response = make_response("Saved Successfully")
        response.headers['Access-Control-Allow-Origin'] = '*' 
        response.content_type = 'application/json'
        return response, 200

def getDiscountRate(event, change_level):
        return 5
        #return get_damage_ratio
        
@app.get('/calculate_sealevel_change_impact')
def calculate_sealevel_change_impact ():
        change_level = request.args.get('seaLevel')
        primary_array = []
        ratio = getDiscountRate("sealevel", change_level)
        # define the name of the CSV file
        analysis_filename = 'portfolio_after_impact_analysis.csv'
        
        # Reading the CSV file
        with open(DATA_FOLDER + 'Portfolio.csv') as csv_file:
                csv_reader = csv.reader(csv_file)
        
        for line in csv_reader:
                # Convert the line to an array
                row = list(line)
                if ".NS" in row[0]: # India Stock - Need to use Ratio here
                        current_price = row[3]
                        future_price_after_discounting = float(current_price) - (float(current_price) * (ratio/100))
                        row[3] = future_price_after_discounting
                        primary_array.append(row)
                else:
                        print(row[0], " Non Indian stock")
                
        # open the CSV file in write mode
        with open(analysis_filename, mode='w', newline='') as file:
                # create a CSV writer object
                writer = csv.writer(file)

        # write the data to the CSV file
        for row in primary_array:
                writer.writerow(row)
        
        response = make_response("Analyzed Successfully")
        response.headers['Access-Control-Allow-Origin'] = '*' 
        response.content_type = 'application/json'
        return response, 200

@app.get('/calculate_temp_change_impact')
def calculate_temp_change_impact ():
        change_level = request.args.get('tempLevel')
        primary_array = []
        ratio = getDiscountRate("templevel", change_level)
        # define the name of the CSV file
        analysis_filename = 'portfolio_after_impact_analysis.csv'
        
        # Reading the CSV file
        with open(DATA_FOLDER + 'Portfolio.csv') as csv_file:
                csv_reader = csv.reader(csv_file)
        
        for line in csv_reader:
                # Convert the line to an array
                row = list(line)
                if ".NS" in row[0]: # India Stock - Need to use Ratio here
                        current_price = row[3]
                        future_price_after_discounting = float(current_price) - (float(current_price) * (ratio/100))
                        row[3] = future_price_after_discounting
                        primary_array.append(row)
                else:
                        print(row[0], " Non Indian stock")
                
        # open the CSV file in write mode
        with open(analysis_filename, mode='w', newline='') as file:
                # create a CSV writer object
                writer = csv.writer(file)

        # write the data to the CSV file
        for row in primary_array:
                writer.writerow(row)
        
        response = make_response("Analyzed Successfully")
        response.headers['Access-Control-Allow-Origin'] = '*' 
        response.content_type = 'application/json'
        return response, 200

@app.route('/add_esg_to_portfolio', methods=['GET'])
def add_esg_to_portfolio():
        dffrom = pd.read_csv('esg_scores.csv', encoding='unicode-escape')
        dfto = pd.read_csv('Portfolio_workshop_draft.csv', encoding='unicode-escape')
        dffrom.set_index('Symbol', inplace=True)
        dfto.set_index('Symbol', inplace=True)
        dfto['ptotalEsg']=dffrom['totalEsg']
        dfto.to_csv('Portfolio.csv')
        return "Total ESG successfully added to Portfolio", 200

@app.route('/add_weightage_to_stocks_in_portfolio', methods=['GET'])
def add_weightage_to_stocks_in_portfolio():
        sumTotalEsg = 0.00
        weightaged_2D_list = [[]]
        with open('Portfolio.csv') as csvfile:
                csvreader = csv.DictReader(csvfile)
                for row in csvreader:
                        preConvertedNum = row['ptotalEsg']
                        if preConvertedNum == '':
                                sumTotalEsg = sumTotalEsg + 0
                                print('-------')
                                print(sumTotalEsg)
                        elif preConvertedNum != '' and preConvertedNum != None:
                                sumTotalEsg = sumTotalEsg + float(row['ptotalEsg'])
                csvfile.close()
        weightage = 0.00
        with open('Portfolio.csv') as csvfile1:
                csvreader1 = csv.DictReader(csvfile1)
                wSymbol = ''
                totalWeightage = 0.00
                for row in csvreader1:
                        wSymbol = row['Symbol']
                        preConvertedNum1 = row['ptotalEsg']
                        if preConvertedNum1 == '':
                                weightage = 0.00
                        elif preConvertedNum1 != '' and preConvertedNum != None:
                                weightage = (float(row['ptotalEsg'])/sumTotalEsg)*100
                        weightaged_2D_list.append([wSymbol , weightage])
                        totalWeightage = totalWeightage + weightage
                print(totalWeightage)

        dffrom = pd.DataFrame(weightaged_2D_list, columns=['Symbol', 'Weightage'])
        dfto = pd.read_csv('Portfolio_workshop_draft.csv', encoding='unicode-escape')
        dffrom.set_index('Symbol', inplace=True)
        dfto.set_index('Symbol', inplace=True)
        dfto['Weightage'] = dffrom['Weightage']
        dfto.to_csv('Portfolio_workshop_draft.csv')
        return jsonify("Weightage successfully added to Portfolio"), 200

@app.route('/get_portfolio_yield', methods=['GET'])
def get_portfolio_yield():
        date_df = open(DATA_FOLDER + 'date_criteria.txt', encoding='unicode-escape')
        snp = pd.read_csv(DATA_FOLDER + 'Portfolio.csv', encoding='unicode-escape')
        snp.set_index('Symbol', inplace=True)
        snp.head()
        tickers = snp.index.to_list()

        px = pd.read_csv(DATA_FOLDER + 'prices.csv')
        px.Date = pd.to_datetime(px.Date)
        px.set_index('Date', inplace=True)
        rows = []
        with open(DATA_FOLDER + 'date_criteria.txt') as dt_file:
                for date_item in dt_file:
                        pv = 0
                        for ticker in tickers:
                                try:
                                        in_date = date_item.strip()
                                        in_pv = 0
                                        if math.isnan(px[ticker][date_item.strip()]):
                                                in_pv = 0
                                        else:
                                                in_pv = (px[ticker][date_item.strip()]*snp['Weightage'][ticker])
                                        pv = pv + in_pv
                                except Exception as e:
                                        print(e)
                                        continue
                        rows.append([date_item.strip(), pv])
        with open('portfolioyield.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(rows)
                file.close()
        
        return jsonify ("success in build porfolio yield"), 200

#@app.get('/get_damage_ratio')
#def get_damage_ratio():
#        # We initialise a dummy ImpactFunc for tropical cyclone wind damage to building.
#        # Giving the ImpactFunc an arbitrary id 3.
#        haz_type = "TC"
#        id = 3
#        name = "TC building damage"
#        # provide unit of the hazard intensity
#        intensity_unit = "m/s"
#        # provide values for the hazard intensity, mdd, and paa
#        intensity = np.linspace(0, 100, num=15)
#        mdd = np.concatenate((np.array([0]), np.sort(np.random.rand(14))), axis=0)
#        paa = np.concatenate((np.array([0]), np.sort(np.random.rand(14))), axis=0)
#        imp_fun = ImpactFunc(
#        id=id,
#        name=name,
#        intensity_unit=intensity_unit,
#        haz_type=haz_type,
#        intensity=intensity,
#        mdd=mdd,
#        paa=paa,
#        )

        # check if the all the attributes are set correctly
#        imp_fun.check()
#        damage_ratio = imp_fun.calc_mdr(18.7)
        
#        return damage_ratio        

        if __name__ == '__main__':
            app.run('0.0.0.0', port=8099)
            app.debug = True           