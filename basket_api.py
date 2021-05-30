import requests,scipy
import time,re
from datetime import datetime
from bs4 import BeautifulSoup
import numpy as np

base_url = "http://basket.co.il/"
stats_individual_base_url = base_url + "stats-individual.asp?lang=en"
player_url = base_url + "player.asp?PlayerId={}&lang=en"
game_url = base_url + "game-zone.asp?GameId={}&lang=en"

reg_exp_player = re.compile("(.*)PlayerId=(.*)&(.*)")
reg_exp_team = re.compile("(.*)TeamId=(.*)&(.*)")
reg_exp_game = re.compile("(.*)GameId=(.*)&(.*)")
reg_game_time =re.compile(r"(\d+/\d+/\d+)")


def parse_table(table_base):
    header = [''.join(tr.text.split()) for tr in table_base.find("tr",{"class":"header_row"}).find_all('td')]

    i = 0
    for tr in table_base.find("tr", {"class": "header_row_2"}).find_all('td'):
        colspan = 1
        if tr.has_attr("colspan"):
            colspan = int(tr["colspan"])
        for j in range(colspan):
            text = tr.text.strip()
            if text!='':
                header[i]+='_'+text
            i+=1

    out_rows = []
    for row in table_base.find_all("tr", {"class": "row even"})+table_base.find_all("tr", {"class": "row odd"}):
        out_row = {}
        for col_name,data in zip(header,row.find_all('td')):

            if col_name == "PlayerName":
                out_row['player_name']    = data.text.strip()
                try:
                    out_row["player_id"] = reg_exp_player.search(data.find('a')['href']).group(2)
                except:
                    out_row["player_id"] = ''

                continue

            if col_name == "Team":
                out_row['team_name']    = data.text.strip()
                out_row["team_id"] = reg_exp_team.search(data.find('a')['href']).group(2)
                continue

            if col_name == "Game":
                out_row['game_name']    = data.text.strip()
                out_row["game_id"]      = reg_exp_game.search(data.find('a')['href']).group(2)
                continue

            if col_name=='SF':
                out_row['SF'] = data.text.strip()=='*'
                continue

            if '/' in data.text and '/' in col_name:
                col_names = [st+'_'+col_name.split('_')[1] for st in col_name.split('_')[0].split('/')]
                col_datas = data.text.strip().split('/')
                for name,data in zip(col_names,col_datas):
                    out_row[name] = eval(data)
                #break amos
                continue #yarin add

            if '%'in data.text:
                out_row[col_name] = eval(data.text.replace('%',''))/100
                continue

            if data.text == '-':
                out_row[col_name] = scipy.nan
                continue

            try:
                out_row[col_name] = eval(data.text.strip())
            except:
                out_row[col_name] = scipy.nan

        out_rows.append(out_row)
    return out_rows


response = requests.get(stats_individual_base_url)
soup = BeautifulSoup(response.text,'html.parser')
years = [year["value"] for year in soup.find('form',{"name":"years"}).find_all('option') if year["value"] != "0"]

all_rows = []
errors = []
for year in years:
    print(year)
    curr_year_url = stats_individual_base_url+"&cYear="+year
    response = requests.get(curr_year_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    urls = [base_url + box_page.find('a')['href'] for box_page in soup.find('div', {"id": "paging_web"}).find_all("div",{"class":"box_page"}) if box_page.find('a') is not None]
    for url in urls:
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            table_base = soup.find("table", {"class": "stats_tbl"})
            all_rows += parse_table(table_base)
        except:
            errors.append(url)
            print(len(errors),' errors')


all_players_rows = []

# yarin add
all_players_ids = np.load('all_players_ids.npy')
all_players_ids[all_players_ids == 115] = 162
all_players_ids[all_players_ids == 131] = 5
problematic = []
problematic = [ 132,   147,  1116,  1135,  1153,  1175,  1188,  1242,  1255,  1256,  1282,  7852,
  8206,  8234,  8261,  8329,  8446,  8449,  8466,  8481,  8572,  8580,  8581,  8869,
  8887,  8908,  8910,  8917,  8953,  8960,  9081,  9108,  9124,  9134,  9145,  9149,
  9158,  9216,  9238,  9307,  9373,  9561,  9578,  9590,  9608,  9711,  9747,  9760,
  9762,  9775,  9776,  9779,  9820,  9823,  9849,  9974, 10020, 10028, 10069, 10090,
 10091, 10105, 10157, 10197, 10219, 10221, 10223, 10252, 10281, 10453, 10484, 10499,
 10515, 10521, 10525, 10537, 10558, 10609, 10627, 10650, 10716, 10791, 10799, 10824,
 10829, 10965, 10991, 11044]
for k in range(len(all_players_ids)): #int(0.68*len(all_players_ids)),
    print(k/len(all_players_ids))
    player_rows = []
    #if all_players_ids[k] ==
    response = requests.get(player_url.format(str(all_players_ids[k])))
    soup = BeautifulSoup(response.text, 'html.parser')
    tables_base = soup.find_all("table", {"class": "stats_tbl"})
    title_string = str(soup.find_all("title")[0])
    f_lim = title_string.rfind('|')
    l_lim = title_string.find('</title>')
    player_name = title_string[f_lim + 2: l_lim]
    if all_players_ids[k] in problematic:
        all_players_ids[k] = int(response.request.url[response.request.url.find('=') + 1:len(response.request.url)])
        response = requests.get(player_url.format(str(all_players_ids[k])))
        soup = BeautifulSoup(response.text, 'html.parser')
        tables_base = soup.find_all("table", {"class": "stats_tbl"})
        title_string = str(soup.find_all("title")[0])
        f_lim = title_string.rfind('|')
        l_lim = title_string.find('</title>')
        player_name = title_string[f_lim + 2: l_lim]
    player_id = str(all_players_ids[k])
    team_names = []
    team_ids = []
    total_team_ids = {}
    cont_flag = False
#    curr_row = parse_table(tables_base)
    for table_base in tables_base:
        if table_base.find('td',{'class':'round_break en'}) is None:
            problematic.append(all_players_ids[k])
            cont_flag = True
            continue
        if ('regular season' in table_base.find('td',{'class':'round_break en'}).text.lower() or 'playoff' in table_base.find('td',{'class':'round_break en'}).text.lower()) and not('stats by games' in table_base.find('td',{'class':'round_break en'}).text.lower()):
            rows = table_base.find_all('td',{'class':'da_ltr_center space'})
            for row in rows:
                if 'TeamId=' in str(row):
                    str_row = str(row)
                    lim1 = str_row.find('TeamId=')
                    str_row_shortened = str_row[lim1:len(str_row)]
                    lim2 = str_row_shortened.find('">')
                    total_team_ids[row.text.lower()] = str_row_shortened[7:lim2]

    if cont_flag:
        continue

    for table_base in tables_base:
        if 'stats by games' in table_base.find('td',{'class':'round_break en'}).text.lower():
            unofficial_name = table_base.find('td',{'class':'round_break en'}).text.lower()
            ff1 = unofficial_name.find('(')
            ff2 = unofficial_name.find(')')
            unofficial_name = unofficial_name[ff1 + 1:ff2]
            unofficial_name = unofficial_name.lower()
            player_rows += parse_table(table_base)
            href_lists = table_base.find_all('td', {'class': 'da_rtl_right reduce space'})

            for j in range(len(href_lists)):
                address_string = str(href_lists[j])
                f_lim = address_string.find('a href="')
                l_lim = address_string.find('lang=en">')
                final_address = address_string[f_lim + 8: l_lim + 7]

                f_lim = address_string.find('lang=en">')
                l_lim = address_string.find('</a>')
                #team_names.append(address_string[f_lim + 9: l_lim])
                final_address = final_address.replace(';', '&')
                final_address = final_address.replace('&amp', '')
                response2 = requests.get('http://basket.co.il/' + final_address)
                soup2 = BeautifulSoup(response2.text, 'html.parser')
                new_tables = soup2.find_all("table", {"class": "stats_tbl"})
                found_flag = False
                for new_table in new_tables:
                    unofficial_name2 = new_table.text.lower()
                    if not(str(new_table).find(player_name) == -1):
                        team_names.append(new_table.find('td', {'class': 'round_break en'}).text.lower())
                        whole_row = str(new_table.find('td',{'class':'round_break en'}))
                        s_lim = whole_row.find('TeamId=')
                        e_lim = whole_row.find('&amp;lang=en')
                        team_id = whole_row[s_lim + 7:e_lim]
                        team_ids.append(team_id)
                        found_flag = True
                    # elif unofficial_name in unofficial_name2:
                    #     team_names.append(new_table.find('td', {'class': 'round_break en'}).text.lower())
                    #     whole_row = str(new_table.find('td', {'class': 'round_break en'}))
                    #     s_lim = whole_row.find('TeamId=')
                    #     e_lim = whole_row.find('&amp;lang=en')
                    #     team_id = whole_row[s_lim + 7:e_lim]
                    #     team_ids.append(team_id)
                if not found_flag:
                    for key in total_team_ids:
                        if unofficial_name.lower() in key.lower():
                            team_ids.append(total_team_ids[key])
                            team_names.append(key)

    count = 0
    for player_row in player_rows:
        player_row['player_name'] = player_name
        player_row['player_id'] = player_id
        player_row['team_name'] = team_names[count]
        player_row['team_id'] = team_ids[count]
        count += 1

    all_players_rows+=player_rows

for i,row in enumerate(all_rows):
    print(i/len(all_rows))
    player_rows = []
    response = requests.get(player_url.format(row['player_id']))
    soup = BeautifulSoup(response.text, 'html.parser')
    tables_base = soup.find_all("table", {"class": "stats_tbl"})
    for table_base in tables_base:
        if 'stats by games' in table_base.find('td',{'class':'round_break en'}).text.lower():
            player_rows += parse_table(table_base)

    for player_row in player_rows:
        for cname in ['player_name','player_id','team_name','team_id']:
            player_row[cname]= row[cname]
    all_players_rows+=player_rows




game_ids = set([player_row['game_id'] for player_row in all_players_rows])

all_games = {}

for i,game_id in enumerate(game_ids):
    print(i/len(game_ids))
    response = requests.get(game_url.format(game_id))
    soup = BeautifulSoup(response.text, 'html.parser')
    tables_base = soup.find_all("table", {"class": "stats_tbl"})

    if len(tables_base)==6 or len(tables_base)==4:
        tables_base = tables_base[2:4]

    if len(tables_base)!=2:
        print('error')
        break

    game = {}
    date_header =  soup.find('div', {'id': 'page_game_zone_header'}).find('h5',{'class':'en'})

    game['datr'] = datetime.strptime(reg_game_time.search(date_header.text).group(0),'%d/%m/%Y')



    game['home_team_id'] = reg_exp_team.search(
        tables_base[0].find('td', {'class': 'round_break en'}).find('a')['href']).group(2)
    game['home_team_name'] = tables_base[0].find('td', {'class': 'round_break en'}).text

    game['away_team_id'] = reg_exp_team.search(
        tables_base[1].find('td', {'class': 'round_break en'}).find('a')['href']).group(2)
    game['away_team_name'] = tables_base[1].find('td', {'class': 'round_break en'}).text


    home_table = parse_table(tables_base[0])
    away_table = parse_table(tables_base[1])

    game['home_team_points'] = [k['Pts'] for k in home_table if k['player_name'].lower()=='total'][0]
    game['away_team_points'] = [k['Pts'] for k in away_table if k['player_name'].lower() == 'total'][0]

    game['home_table'] = home_table
    game['away_table'] = away_table
    all_games[game_id] = game

import pickle
save_dict = {'games': all_games, 'player_games': all_players_rows, 'player_stats': all_rows}
with open('basket_data.pkl','wb+') as file:
    pickle.dump(save_dict,file)

problematic = np.array(problematic)
problematic = np.unique(problematic)
print(problematic)
np.save('problematic.npy', problematic)
# with open('basket_data.pkl', 'rb') as file:
#     save_dict = pickle.load(file)


x ={"dates": [0, 2, 5, 10, 12, 15, 17, 20, 22, 25, 30, 37, 47, 52, 55, 65, 70, 80, 82, 85, 87, 92, 95, 97, 102, 105, 107, 112, 115, 117, 120, 122, 127, 130, 135, 137, 140, 142, 145, 147, 150, 152, 157, 162, 170, 177, 180, 185, 187, 195, 200, 210, 215, 220, 227, 232, 237, 240, 260, 265, 267, 272, 280, 282, 295, 302, 305, 310, 312, 320, 325, 332, 337, 340, 345, 352, 357, 362, 365, 367, 370, 380, 382, 385, 387, 390, 395, 402, 407, 412, 415, 417, 420, 422, 425, 427, 435, 440, 445, 452, 455, 460, 462, 467, 470, 480, 482, 485, 487, 490, 492, 500, 502, 505, 507, 510, 512], "is_growing": [0.8290557788459806, 0.8658794951901462, 0.9587802921516027, 0.7112226282675039, 0.5913834196038372, 0.5167359801704379, 0.5050170010406259, 0.5371079324947143, 1.1040516478306426, 1.0723960115447133, 0.8641638141823776, 0.7716568215953885, 0.731456033745663, 1.1166835377916808, 1.0918685762898788, 1.013290060659951, 0.8476484913094168, 0.5093649769012544, -0.030610803792431782, 0.2336271277675722, 0.19623835949346893, 0.20524733723143912, 0.1769071497280321, 0.1648906918527068, 0.19729491022037363, 0.19440422253946152, 0.14991640457929184, 0.1500195268981971, -0.012607844385005765, 0.10446209623601523, 0.11171193212876149, 0.11437564433117461, 0.15225302905678156, 0.12948766760289998, 0.11534227303467498, 0.09502826520743499, 0.09434101315202864, 0.10026578763993212, 0.059537997375492833, 0.08505299893920544, 0.08289357944747755, 0.0680823856852551, -0.020809357526442862, -0.05326607674818122, 0.0975481851979646, 0.17042299471263225, 0.036148088007395114, 0.14878339234435045, -0.05612216526537194, 0.37636428505837155, 0.49796957285543103, 0.3871945244814757, 0.7091594408918837, 0.6647586981003756, 0.6118847299385394, 0.6663303755047484, 0.6961486754741188, 0.891280254397552, 0.4832848966098317, 0.4835894695795783, 0.4760801512121831, 0.2771239374608497, 0.264088510077969, 0.31527561353721817, 0.0729132055272077, 0.1758211357976568, -0.015790027863372233, 0.20374107465309826, 0.09643593704858683, 0.04909214493264209, 0.08158840535533873, 0.10152675647105337, 0.14083142913286842, 0.1638292879731859, 0.18435784645914335, 0.19922884317606016, 0.2867309935565588, 0.024644913717143, 0.30122536068623995, 0.4143727032057569, 0.3393138877244543, 0.6081965095536027, 0.585731404599692, 0.6036780828207009, 0.8779354230098162, 0.7405511798329755, 0.6435873027587965, 0.5101749706851704, 1.172265282982732, 0.803662455101325, 0.5638711060508331, 0.5370578019612908, 1.051674803639135, 1.15768053158734, 0.8716254113007286, 0.6956524881172994, 1.1249677252380488, 0.6086548630339226, 0.8737555505901345, 0.5627003762062162, 0.118064208816606, 0.06456989912577406, 0.23548176842113816, 0.18810418416935476, 0.20929896813053453, 0.19528443594165903, 0.17454042829291624, -0.06381739644304592, 0.14022453889568764, 0.19246697000256766, 0.1658683426390403, 0.11789673985403538, 0.12910370166494708, 0.12232620461030315, 0.11980751754924501, 0.12790429371287654, 0.09894834511998482], "ndvis": [0.5366906474820143, 0.5585305105853051, 0.6136292591434823, 0.6414831335377753, 0.05780476833108412, 0.04498403792202767, 0.041147721898155444, 0.14777327935222673, 0.6997885835095138, 0.681013868962219, 0.5575129533678757, 0.5660722450845908, 0.32093362509117435, 0.7072804603452589, 0.6925628974288084, 0.6459585838343354, 0.5477178423236515, 0.3470844902816343, 0.02682900549509751, 0.1835464915312824, 0.16137150047184648, 0.16671465591707457, 0.14990630855715179, 0.14277943976067448, 0.1619981325863679, 0.16028368794326242, 0.13389830508474576, 0.13395946613939694, 0.037506422332591195, 0.10693970420932879, 0.11123952614851199, 0.11281935338005879, 0.13528413910093298, 0.12178217821782178, 0.1133926534337212, 0.10134457154324704, 0.10093696763202725, 0.10445090583762022, 0.08029556650246306, 0.0954283177984909, 0.09414758269720101, 0.08536317752907614, 0.0326421679326627, 0.013392337253988588, 0.1028391167192429, 0.14606060606060606, 0.06642319105227117, 0.1332263242375602, 0.011698413108925774, 0.2734135409294645, 0.5142857142857142, 0.059318885448916406, 0.4655810510732791, 0.439247311827957, 0.40788816774837744, 0.44017946161515453, 0.45786446611652915, 0.5735955056179776, 0.33161659513590847, 0.3317972350230415, 0.3273435160227613, 0.20934411500449238, 0.20161290322580644, 0.23197158081705152, 0.08822829964328181, 0.1492622020431328, 0.03561909377271626, 0.16582130316035895, 0.10217945089159354, 0.07410021171489062, 0.09337349397590361, 0.10519877675840979, 0.12851007598282127, 0.14214992927864215, 0.15432525951557094, 0.16314513335642536, 0.21504198612632347, 0.05960074680453827, 0.18952760387023335, 0.45589798087141337, 0.06165764618486681, 0.40570071258907364, 0.6440586001085187, 0.014462809917355372, 0.5656807887023715, 0.48419925013390464, 0.598371086790532, 0.025821456258572364, 0.7402455661664393, 0.698636926889715, 0.01536627367863293, 0.20873891412502704, 0.668724279835391, 0.7315954622254405, 0.638076351016361, 0.24000531279054324, 0.7121937482399324, 0.4059725585149314, 0.5632017385005433, 0.37871730562522393, 0.1150070126227209, 0.08327994875080078, 0.18464646464646464, 0.15654718361375275, 0.16911764705882354, 0.1608057357459884, 0.14850262426674898, 0.007134430649963486, 0.12815013404825737, 0.15913471835510817, 0.1433592769269395, 0.11490768806833838, 0.12155445097153186, 0.11753476956640306, 0.11604095563139932, 0.12084309133489461, 0.10366954080609585], "start_date": "11-01-2018"}