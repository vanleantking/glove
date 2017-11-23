import re
import random
import string
import sys
from faker import Faker
from datetime import datetime
from calendar import monthrange

VOWELS = "aeiou"
CONSONANTS = "".join(set(string.ascii_lowercase) - set(VOWELS))


class FakerInfo:

    date_patterns = ["%m-%d-%y", "%Y-%m-%d", "%d-%b-%y", "%d-%b-%Y", "%m-%d", "%m-%y", "%b-%Y",
                    "%m/%d/%y", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d", "%m/%d", "%m/%y", "%m/%Y",
                    "%m.%d.%y", "%d.%m.%y", "%b. %d", "%B. %d", "%b. %y", "%b. %Y", "%B. %y", "%B. %Y",
                    "%d %b %Y", "%d %B %Y", "%B %d %Y", "%m %Y", "%b %d", "%B %d", "%b %y", "%b %Y", "%B %Y", "%B %y", "%d %B",
                    "%b %d, %Y", "%B %d, %Y", "%A, %B %d, %Y", "%b, %Y", "%A, %B %d", "%B, %Y", "%b, %Y", 
                    "%a", "%A", "%b", "%B", "%m", "%d", "%y", "%Y"]
    date_decade = '\d+\s*\'*s'
    seasons = ['spring', 'summer', 'autumn', 'winter']
    weekday = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
    normalize_weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    holidays = {"thanksgiving": "15 August", "ramada": "15 May", "easter": "1 April"}
    symbol = ['\'', 'of']
    ages_word = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    ages_number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    day_shift = 3
    date_shift = 3
    month_shift = 3
    year_shift = 5
    season_shif = 1

    def __init__(self):
        self._faker = Faker()

    def get_faker(self):
        return self._faker

    def generate_name(self):
        return self._faker.name()

    def generate_username(self, size=6, chars=string.ascii_uppercase + string.digits):
        return ''.join(random.SystemRandom().choice(chars) for _ in range(size))

    def generate_phone(self):
        return self._faker.phone_number()

    def generate_fax(self):
        return self._faker.name()

    def generate_email(self):
        return self._faker.name()

    def generate_URL(self):
        return self._faker.name()

    def generate_ID(self):
        return self._faker.name()

    def generate_hospital(self):
        return self._faker.company()

    def generate_city(self):
        return self._faker.city()

    def generate_state(self):
        return self._faker.state()

    def generate_street(self, streets):
        street_number = []
        faker_street = self._faker.street_name()
        for street in streets:
            tmp = re.findall(r'(\d+)', street.name)
            if len(tmp) > 0:
                for number in tmp:
                    street_number.append(number)
            else:
                street_number.append("")
        return [number + ' ' + faker_street for number in street_number]


    def generate_zip(self):
        return self._faker.zipcode_plus4()

    def generate_company(self):
        return self._faker.company()

    def generate_country(self):
        return self._faker.country()

    def generate_profession(self):
        return self._faker.job()

    def generate_date(self, strdate):

        date_process = strdate.strip().lower()
        #check date is decade
        if re.match(self.date_decade, date_process):
            return strdate

        #check date is day:
        week_day = date_process.split()
        if len(week_day) == 1:
            for index, day in enumerate(self.weekday):
                if re.match(day, date_process):
                    return self.normalize_weekdays[index + self.day_shift - len(self.weekday)] if index + self.day_shift >= len(self.weekday) else self.weekday[index + self.day_shift]

        #check for season only
        for index, season in enumerate(self.seasons):
            if season == date_process:
                return self.seasons[index + self.season_shif - len(self.seasons)] if index + self.season_shif >= len(self.seasons) else self.seasons[index + self.season_shif]

        #check for season and year:
        season_year = re.findall(r"[\w']+|[.,!?;\/+]", date_process)
        for index, season in enumerate(self.seasons):
            for idx, word in enumerate(season_year):
                if word == season:
                    season_year[idx] = self.seasons[idx + self.season_shif - len(self.seasons)] if idx + self.season_shif >= len(self.seasons) else self.seasons[idx + self.season_shif]
                    season_year[-1] = str(int(season_year[-1]) + self.year_shift)
                    return " ".join(season_year)



        d_obj = None

        #check holidays and year
        h_flag = False
        index_flag = 0
        holiday_year = re.findall(r"[\w']+|[.,!?;\/+]", date_process)
        for index, word in enumerate(holiday_year):
            if word in self.holidays:
                h_flag = True
                break

        #convert holiday to a specific day
        if h_flag == True:
            holiday_year[index] =  self.holidays[word]
            reg_date = " ".join(holiday_year)
        
        #remove special symbols in date string 
        else:
            for sym in self.symbol:
                date_process = date_process.replace(sym, "")

            date_process = re.sub(' +',' ',date_process)
            reg_date = re.sub(r'(\d)(st|nd|rd|th)', r'\1', date_process)
        for pattern in self.date_patterns:
            try:
                d_obj = datetime.strptime(reg_date, pattern)
                p = pattern
                break
            except:
                d_obj = None
        if d_obj is None:
            return strdate

        f_m = (d_obj.month + self.month_shift - 12) if (d_obj.month + self.month_shift) > 12 else (d_obj.month + self.month_shift)
        f_y = int('20' + str(d_obj.year + self.year_shift)[2:]) + 100 if (d_obj.year - 2000) >= 100 else 2000 + int(str(d_obj.year)[2:]) + self.year_shift
        
        min_day, max_day = monthrange(f_y, f_m)
        if (d_obj.day + self.day_shift) > max_day:
            f_d = d_obj.day + self.day_shift - max_day
            f_m = f_m + 1 - 12 if (f_m + 1) > 12 else f_m + 1
        else:
            f_d = d_obj.day + self.day_shift

        return datetime.strptime(str(f_m) + '/' + str(f_d) + '/' + str(f_y), '%m/%d/%Y').strftime(p), f_y


    def generate_age(self, age):
        if re.match(self.date_decade, age):
            return age
        age_process = re.findall(r"[\d]+|[\w]+|[,.]", age)
        if len(age_process) == 1:
            try:
                if int(age) > 10:
                    return age[0] + str(self.get_random(int(age[1]), 9, 0))
                else:
                    return str(self.get_random(int(age), 9, 0))
            except:
                return age
        else:
            try:
                if int(age_process[0]) > 10:
                    age_process[0] = str(int(age_process[0][0] + str(self.get_random(int(age_process[0][1]), 9, 0))))
                else:
                    age_process[0] = str(self.get_random(int(age_process[0]), 9))
            except:
                age_process[-1] = random.choice(self.ages_word)
            
        return " ".join(age_process)


    def get_random(self, n, end, start = 1):
        return random.choice(list(range(start, n)) + list(range(n+1, end)))

if __name__ == "__main__":
    fake = FakerInfo()

    print(fake.generate_age('13 month'))
    print(fake.generate_age('15\'s'))
    print(fake.generate_age('15      \'s'))
    print(fake.generate_age('25s'))
    print(fake.generate_age('13y7.7m'))
    print(fake.generate_age('6 weeks'))
    print(fake.generate_age('6weeks'))
    print(fake.generate_age('6mos'))
    print(fake.generate_age('6 mos'))
    print(fake.generate_age('45'))
    print(fake.generate_age('5'))
    print(fake.generate_age('twenty four'))
    print(fake.generate_age('twenty-four'))
    print(fake.generate_age('twenty'))
    print(isinstance('45', int))

    # fake = Faker()

    # print(fake.name())
    # print(fake.street_address())
    # print(fake.street_name())
    
    # for pattern in FakerInfo.date_patterns:
    #     try:
    #         print(datetime.strptime('13/2/2132', pattern).date())
    #         print(pattern)
    #         break
    #     except:
    #         pass
    # print(d_obj.year)

    # n = '2013'
    # print(n[2:])
        # except:
        #     print('sdfds')

    # try:
    #     count = int(sys.argv[1])
    # except:
    #     count = 5

    # try:
    #     length = int(sys.argv[2])
    # except:
    #     length = 6

    # for i in range(count):
    #     print(generate_word(length))