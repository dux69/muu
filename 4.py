import os
import requests
import logging
import re
from telegram import Update
from telegram.ext import Updater, CommandHandler, CallbackContext

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

TOKEN = '6053880139:AAH4KR4A7RTzkAmrnFkbnBNAGA3vrYV9Jes'

GPUS = {
    'rtx3090': {'name': 'RTX 3090', 'hashrate_rvn': 0.06, 'hashrate_etc': 0.12, 'hashrate_kas': 0.12, 'hashrate_ethw': 0.12, 'hashrate_cfx': 0.12, 'hashrate_erg': 0.12, 'power_consumption_rvn': 350, 'power_consumption_etc': 330, 'power_consumption_kas': 330, 'power_consumption_ethw': 330, 'power_consumption_cfx': 330, 'power_consumption_erg': 330},
    'rtx3080': {'name': 'RTX 3080', 'hashrate_rvn': 0.05, 'hashrate_etc': 0.10, 'hashrate_kas': 0.12, 'hashrate_ethw': 0.12, 'hashrate_cfx': 0.12, 'hashrate_erg': 0.12, 'power_consumption_rvn': 320, 'power_consumption_etc': 300, 'power_consumption_kas': 330, 'power_consumption_ethw': 330, 'power_consumption_cfx': 330, 'power_consumption_erg': 330},
    'rx580': {'name': 'RX 580', 'hashrate_rvn': 0.015, 'hashrate_etc': 0.03, 'hashrate_kas': 0.12, 'hashrate_ethw': 0.12, 'hashrate_cfx': 0.12, 'hashrate_erg': 0.12, 'power_consumption_rvn': 185, 'power_consumption_etc': 165, 'power_consumption_kas': 330, 'power_consumption_ethw': 330, 'power_consumption_cfx': 330, 'power_consumption_erg': 330},
    'rx570': {'name': 'RX 570', 'hashrate_rvn': 0.014, 'hashrate_etc': 0.028, 'hashrate_kas': 0.12, 'hashrate_ethw': 0.12, 'hashrate_cfx': 0.12, 'hashrate_erg': 0.12, 'power_consumption_rvn': 150, 'power_consumption_etc': 140, 'power_consumption_kas': 330, 'power_consumption_ethw': 330, 'power_consumption_cfx': 330, 'power_consumption_erg': 330},
}

USD_PLN = 4.36

def convert_currency(amount: float, from_currency: str, to_currency: str) -> float:
    if from_currency == to_currency:
        return amount

    if from_currency == "USD" and to_currency == "PLN":
        return amount * USD_PLN
    elif from_currency == "PLN" and to_currency == "USD":
        return amount / USD_PLN
    else:
        raise ValueError("Unsupported currency conversion")
        
def handle_dm_start(update: Update, context: CallbackContext):
    user_id = update.message.from_user.id  # Pobierz identyfikator czatu użytkownika
    message = "Hej, oto polecenia które możesz wykonać:"
    context.bot.send_message(chat_id=user_id, text=message)  # Wyślij wiadomość do użytkownika  

def get_cfx_data():
    response = requests.get("https://whattomine.com/coins.json")
    data = response.json()["coins"]["Conflux"]

    block_reward = data["block_reward"]
    nethash = data["nethash"]
    block_reward24 = data["block_reward24"]
    block_time = float(data["block_time"])  # Rzutuj block_time na float

    nethash_ghs = nethash / 1_000_000_000
    return block_reward, nethash_ghs, block_reward24, block_time
    
def get_cfx_price():
    url = "https://api.minerstat.com/v2/coins"
    response = requests.get(url)
    data = response.json()

    for coin_data in data:
        if coin_data.get("coin") == "CFX":
            return coin_data.get("price")
            
def get_ergo_data():
    response = requests.get("https://whattomine.com/coins.json")
    data = response.json()["coins"]["Ergo"]

    block_reward = data["block_reward"]
    nethash = data["nethash"]
    block_reward24 = data["block_reward24"]
    block_time = float(data["block_time"])  # Rzutuj block_time na float

    nethash_ghs = nethash / 1_000_000_000
    return block_reward, nethash_ghs, block_reward24, block_time
    
def get_ergo_price():
    url = "https://api.minerstat.com/v2/coins"
    response = requests.get(url)
    data = response.json()

    for coin_data in data:
        if coin_data.get("coin") == "ERG":
            return coin_data.get("price")            

def get_ethw_data():
    response = requests.get("https://whattomine.com/coins.json")
    data = response.json()["coins"]["EthereumPoW"]

    block_reward = data["block_reward"]
    nethash = data["nethash"]
    block_reward24 = data["block_reward24"]
    block_time = float(data["block_time"])  # Rzutuj block_time na float

    nethash_ghs = nethash / 1_000_000_000
    return block_reward, nethash_ghs, block_reward24, block_time
    
def get_ethw_price():
    url = "https://api.minerstat.com/v2/coins"
    response = requests.get(url)
    data = response.json()

    for coin_data in data:
        if coin_data.get("coin") == "2MINERS ETHW":
            return coin_data.get("price")
            
def get_kaspa_data():
    response = requests.get("https://whattomine.com/coins.json")
    data = response.json()["coins"]["Kaspa"]

    block_reward = data["block_reward"]
    nethash = data["nethash"]
    block_reward24 = data["block_reward24"]
    block_time = float(data["block_time"])  # Rzutuj block_time na float

    nethash_ghs = nethash / 1_000_000_000
    return block_reward, nethash_ghs, block_reward24, block_time
    
def get_kaspa_price():
    url = "https://api.minerstat.com/v2/coins"
    response = requests.get(url)
    data = response.json()

    for coin_data in data:
        if coin_data.get("coin") == "KAS":
            return coin_data.get("price")    
    
def get_ravencoin_data():
    response = requests.get("https://whattomine.com/coins.json")
    data = response.json()["coins"]["Ravencoin"]

    block_reward = data["block_reward"]
    nethash = data["nethash"]
    block_reward24 = data["block_reward24"]

    nethash_ghs = nethash / 1_000_000_000
    return block_reward, nethash_ghs, block_reward24

def get_rvn_price():
    url = "https://api.minerstat.com/v2/coins"
    response = requests.get(url)
    data = response.json()

    for coin_data in data:
        if coin_data.get("coin") == "2MINERS RVN":
            return coin_data.get("price")

def get_etc_data():
    response = requests.get("https://whattomine.com/coins.json")
    data = response.json()["coins"]["EthereumClassic"]

    block_reward = data["block_reward"]
    nethash = data["nethash"]
    block_reward24 = data["block_reward24"]
    block_time = float(data["block_time"])  # Rzutuj block_time na float

    nethash_ghs = nethash / 1_000_000_000
    return block_reward, nethash_ghs, block_reward24, block_time


def get_etc_price():
    url = "https://api.minerstat.com/v2/coins"
    response = requests.get(url)
    data = response.json()

    for coin_data in data:
        if coin_data.get("coin") == "2MINERS ETC":
            return coin_data.get("price")

def start(update: Update, context: CallbackContext):
    update.message.reply_text("Witaj w bot-calc! Poniżej znajdziesz dostępne komendy:\n\n/hashrate <hashrate>\n/gpu <model> <ilość>\n")

def handle_hashrate(update: Update, context: CallbackContext):
    try:
        coin = context.args[0].upper()
        hashrate = float(context.args[1])
        power_consumption = float(context.args[2]) / 1000  # Przelicz zużycie energii na kW
        try:
            electricity_cost = float(context.args[3])  # Próbuj pobrać cenę energii
        except IndexError:
            electricity_cost = 1  # Ustaw domyślną cenę energii na 1
    except (IndexError, ValueError):
        update.message.reply_text("Wprowadź wartość hashrate używając /hashrate <coin> <hashrate> <power_consumption> <electricity_cost>")
        return

    if coin not in ["RVN", "ETC", "KAS", "ETHW", "CFX", "ERG"]:
        update.message.reply_text("Wprowadź poprawny coin (RVN, ETC, KAS, ETHW, CFX, ERG).")
        return
    try:
        currency = context.args[4].upper()
    except IndexError:
        currency = "PLN"

    if currency not in ["USD", "PLN"]:
        update.message.reply_text("Wprowadź poprawną walutę (USD lub PLN).")
        return
       
    if hashrate > 1:
        hashrate /= 1000

    if coin == "RVN":
        block_reward, nethash_ghs, block_reward24 = get_ravencoin_data()
        coin_price = get_rvn_price()
    if coin == "ETC":
        block_reward, nethash_ghs, block_reward24, block_time = get_etc_data()
        coin_price = get_etc_price()
    if coin == "KAS":
        block_reward, nethash_ghs, block_reward24, block_time = get_kaspa_data()
        coin_price = get_kaspa_price()
    if coin == "ETHW":
        block_reward, nethash_ghs, block_reward24, block_time = get_ethw_data()
        coin_price = get_ethw_price()
    if coin == "CFX":
        block_reward, nethash_ghs, block_reward24, block_time = get_cfx_data()
        coin_price = get_cfx_price()
    if coin == "ERG":
        block_reward, nethash_ghs, block_reward24, block_time = get_ergo_data()
        coin_price = get_ergo_price()           

    if coin in ["ETC", "KAS", "ETHW", "CFX", "ERG"]:
        profit_daily = (hashrate * block_reward * 86400) / (nethash_ghs * block_time)
    else:
        profit_daily = (hashrate * block_reward * 86400) / (nethash_ghs * 60)

    profit_daily_usd = profit_daily * coin_price
    profit_weekly = profit_daily * 7
    profit_weekly_usd = profit_weekly * coin_price
    profit_monthly = profit_daily * 30
    profit_monthly_usd = profit_monthly * coin_price

    # Oblicz koszty energii
    daily_energy_cost = power_consumption * electricity_cost * 24
    weekly_energy_cost = daily_energy_cost * 7
    monthly_energy_cost = daily_energy_cost * 30

    # Konwertuj wyniki do żądanej waluty
    profit_daily_converted = convert_currency(profit_daily_usd, "USD", currency)
    profit_weekly_converted = convert_currency(profit_weekly_usd, "USD", currency)
    profit_monthly_converted = convert_currency(profit_monthly_usd, "USD", currency)

    # Oblicz zarobek netto
    net_profit_daily_converted = profit_daily_converted - daily_energy_cost
    net_profit_weekly_converted = profit_weekly_converted - weekly_energy_cost
    net_profit_monthly_converted = profit_monthly_converted - monthly_energy_cost
    
    update.message.reply_text(f"Twoja estymowana rentowność dla {coin} ({currency}):\n\n24h: {profit_daily:.2f} {coin} / {profit_daily_converted:.2f} {currency} (netto: {net_profit_daily_converted:.2f} {currency})\n7dni: {profit_weekly:.2f} {coin} / {profit_weekly_converted:.2f} {currency} (netto: {net_profit_weekly_converted:.2f} {currency})\n30dni: {profit_monthly:.2f} {coin} / {profit_monthly_converted:.2f} {currency} (netto: {net_profit_monthly_converted:.2f} {currency})")

def handle_gpu(update: Update, context: CallbackContext):
    try:
        gpu_model = context.args[0]
        gpu_count = int(context.args[1])
        coin = context.args[2].upper()
        electricity_cost = float(context.args[3])
    except (IndexError, ValueError):
        update.message.reply_text("Wprowadź nazwę modelu GPU, ilość, coin i cenę za prąd używając /gpu <model> <ilość> <coin> <cena_za_prąd>")
        return

    if gpu_model not in GPUS:
        update.message.reply_text("Nieprawidłowy model GPU. Dostępne modele to: " + ", ".join(GPUS.keys()))
        return

    if coin not in ["RVN", "ETC", "KAS", "ETHW", "CFX", "ERG"]:
        update.message.reply_text("Wprowadź poprawny coin (RVN, ETC, KAS, ETHW, CFX, ERG).")
        return
        
    currency = "PLN"

    if currency not in ["USD", "PLN"]:
        update.message.reply_text("Wprowadź poprawną walutę (USD lub PLN).")
        return

    gpu_hashrate = GPUS[gpu_model][f"hashrate_{coin.lower()}"]
    total_hashrate = gpu_hashrate * gpu_count
    power_consumption = GPUS[gpu_model][f'power_consumption_{coin.lower()}'] * gpu_count

    if coin == "RVN":
        block_reward, nethash_ghs, block_reward24 = get_ravencoin_data()
        coin_price = get_rvn_price()
    if coin == "ETC":
        block_reward, nethash_ghs, block_reward24, block_time = get_etc_data()
        coin_price = get_etc_price()
    if coin == "KAS":
        block_reward, nethash_ghs, block_reward24, block_time = get_kaspa_data()
        coin_price = get_kaspa_price()
    if coin == "ETHW":
        block_reward, nethash_ghs, block_reward24, block_time = get_ethw_data()
        coin_price = get_ethw_price()
    if coin == "CFX":
        block_reward, nethash_ghs, block_reward24, block_time = get_cfx_data()
        coin_price = get_cfx_price()
    if coin == "ERG":
        block_reward, nethash_ghs, block_reward24, block_time = get_ergo_data()
        coin_price = get_ergo_price()           

    if coin in ["ETC", "KAS", "ETHW", "CFX", "ERG"]:
        profit_daily = (total_hashrate * block_reward * 86400) / (nethash_ghs * block_time)
    else:
        profit_daily = (total_hashrate * block_reward * 86400) / (nethash_ghs * 60)

    profit_daily_usd = profit_daily * coin_price
    profit_weekly = profit_daily * 7
    profit_weekly_usd = profit_weekly * coin_price
    profit_monthly = profit_daily * 30
    profit_monthly_usd = profit_monthly * coin_price

    # Oblicz koszty energii
    daily_energy_cost = (power_consumption * electricity_cost * 24) / 1000
    weekly_energy_cost = daily_energy_cost * 7
    monthly_energy_cost = daily_energy_cost * 30

    # Konwertuj wyniki do żądanej waluty
    profit_daily_converted = convert_currency(profit_daily_usd, "USD", currency)
    profit_weekly_converted = convert_currency(profit_weekly_usd, "USD", currency)
    profit_monthly_converted = convert_currency(profit_monthly_usd, "USD", currency)

    # Oblicz zarobek netto
    net_profit_daily_converted = profit_daily_converted - daily_energy_cost
    net_profit_weekly_converted = profit_weekly_converted - weekly_energy_cost
    net_profit_monthly_converted = profit_monthly_converted - monthly_energy_cost
    
    update.message.reply_text(f"Twoja estymowana rentowność dla {coin} ({currency}):\n\n24h: {profit_daily:.8f} {coin} / {profit_daily_converted:.2f} {currency} (netto: {net_profit_daily_converted:.2f} {currency}) | \n7dni: {profit_weekly:.8f} {coin} / {profit_weekly_converted:.2f} {currency} (netto: {net_profit_weekly_converted:.2f} {currency}) | \n30dni: {profit_monthly:.8f} {coin} / {profit_monthly_converted:.2f} {currency} (netto: {net_profit_monthly_converted:.2f} {currency})")

def get_net_profit(gpu_model, num_gpus, electricity_cost):
    coins = ['rvn', 'etc', 'kas', 'ethw', 'cfx', 'erg']
    coin_net_profits = {}

    for coin in coins:
        hashrate = GPUS[gpu_model]['hashrate_' + coin] * num_gpus
        power_consumption = GPUS[gpu_model]['power_consumption_' + coin] * num_gpus / 1000
        handle_hashrate_data = [coin.upper(), hashrate, power_consumption, electricity_cost, "PLN"]
        _, _, _, _, _, _, _, net_profit_daily_converted, _, _, _ = handle_hashrate_calc(handle_hashrate_data)
        coin_net_profits[coin] = net_profit_daily_converted

    return coin_net_profits
    
def top5_coins(update: Update, context: CallbackContext):
    try:
        gpu_model = context.args[0].lower()
        gpu_count = int(context.args[1])
        electricity_cost = float(context.args[2])
        display_currency = context.args[3].upper() if len(context.args) > 3 else "USD"
    except (IndexError, ValueError):
        update.message.reply_text("Użyj /top5 <model_gpu> <ilość> <cena_za_prąd> [waluta (USD lub PLN)]")
        return

    if gpu_model not in GPUS:
        update.message.reply_text("Wprowadź poprawny model GPU (rtx3090, rtx3080, rx580, rx570).")
        return

    if display_currency not in ["USD", "PLN"]:
        update.message.reply_text("Nieobsługiwana waluta, wprowadź USD lub PLN.")
        return

    coin_symbols = ["rvn", "etc", "kas", "ethw", "cfx", "erg"]
    coin_profits = []

    for coin_symbol in coin_symbols:
        hashrate_key = f"hashrate_{coin_symbol}"
        power_key = f"power_consumption_{coin_symbol}"
        hashrate = GPUS[gpu_model][hashrate_key] * gpu_count
        power_consumption = GPUS[gpu_model][power_key] * gpu_count / 1000

        if coin_symbol == "rvn":
            block_reward, nethash_ghs, block_reward24 = get_ravencoin_data()
            coin_price = get_rvn_price()
        elif coin_symbol == "etc":
            block_reward, nethash_ghs, block_reward24, block_time = get_etc_data()
            coin_price = get_etc_price()
        elif coin_symbol == "kas":
            block_reward, nethash_ghs, block_reward24, block_time = get_kaspa_data()
            coin_price = get_kaspa_price()
        elif coin_symbol == "ethw":
            block_reward, nethash_ghs, block_reward24, block_time = get_ethw_data()
            coin_price = get_ethw_price()
        elif coin_symbol == "cfx":
            block_reward, nethash_ghs, block_reward24, block_time = get_cfx_data()
            coin_price = get_cfx_price()
        elif coin_symbol == "erg":
            block_reward, nethash_ghs, block_reward24, block_time = get_ergo_data()
            coin_price = get_ergo_price()

        if coin_symbol in ["etc", "kas", "ethw", "cfx", "erg"]:
            profit_daily = (hashrate * block_reward * 86400) / (nethash_ghs * block_time)
        else:
            profit_daily = (hashrate * block_reward * 86400) / (nethash_ghs * 60)
        

        profit_daily_usd = profit_daily * coin_price

        if display_currency == "PLN":
            electricity_cost_usd = convert_currency(electricity_cost, "PLN", "USD")
        else:
            electricity_cost_usd = electricity_cost

        daily_energy_cost = power_consumption * electricity_cost_usd * 24
        net_profit_daily_usd = profit_daily_usd - daily_energy_cost

        net_profit_daily_display_currency = convert_currency(net_profit_daily_usd, "USD", display_currency)

        coin_profits.append((coin_symbol.upper(), net_profit_daily_display_currency))
    
    # Sortuj wyniki według rentowności netto
    coin_profits.sort(key=lambda x: x[1], reverse=True)

    top5_coins_message = f"Top 5 najbardziej opłacalnych coinów (w {display_currency}):\n"
    for i, (coin, net_profit) in enumerate(coin_profits[:5]):
        top5_coins_message += f"{i + 1}. {coin}: {net_profit:.2f} {display_currency}/dzień\n"

    update.message.reply_text(top5_coins_message)


def main():
    updater = Updater(TOKEN, use_context=True)

    dispatcher = updater.dispatcher
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("hashrate", handle_hashrate))
    dispatcher.add_handler(CommandHandler("gpu", handle_gpu))
    dispatcher.add_handler(CommandHandler("dmstart", handle_dm_start))  # Dodajemy nową komendę dmstart
    dispatcher.add_handler(CommandHandler("top5", top5_coins))


    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()
