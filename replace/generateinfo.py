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
                    "%m/%d/%y", "%m/%d/%Y", "%Y/%m/%d", "%m/%d", "%m/%y", "%m/%Y",
                    "%m.%d.%y", "%b. %y", "%b. %Y",
                    "%d %b %Y", "%m %Y", "%b %d", "%B %d", "%b %y", "%b %Y", "%B %Y", "%B %y",
                    "%b %d, %Y", "%B %d, %Y", "%A, %B %d, %Y", "%b, %Y", "%A, %B %d", "%B, %Y", "%b, %Y", 
                    "%a", "%A", "%b", "%B", "%m", "%d", "%y", "%Y"]
    date_decade = ['\'s', '\d+\s*s']
    seasons = ['spring', 'summer', 'autumn', 'winter']
    weekday = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
    normalize_weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    holidays = {"thanksgiving": "15 August", "ramada": "15 May", "easter": "1 April"}
    symbol = ['\'', 'of']
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

    def generate_username(self):
        return self._faker.name()

    def generate_age(self, ages):
        return self._faker.name()

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
        return self._faker.name()

    def generate_city(self):
        return self._faker.city()

    def generate_state(self):
        return self._faker.state()

    def generate_street(self, streets):
        faker_street = self._faker.street_name()
        street_number = [number for street in streets for number in re.split(r'(\d+)', street.name) if number]
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
        for decade in self.date_decade:
            if re.match(decade, date_process):
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
                    word = self.seasons[idx + self.season_shif - len(self.seasons)] if idx + self.season_shif >= len(self.seasons) else self.seasons[idx + self.season_shif]
                    season_year[-1] = str(int(season_year[-1]) + self.year_shift)
                    return " ".join(season_year)



        d_obj = None
        if date_process in self.holidays:
            reg_date = self.holidays[date_process]

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
            f_m += 1
        else:
            f_d = d_obj.day + self.day_shift

        return datetime.strptime(str(f_m) + '/' + str(f_d) + '/' + str(f_y), '%m/%d/%Y').strftime(p), f_y




if __name__ == "__main__":

    fake = Faker()

    # print(fake.name())
    # print(fake.street_address())
    # print(fake.street_name())
    
    # for pattern in FakerInfo.date_patterns:
    #     try:
            # print(datetime.strptime('29-03-78', pattern).date())
    d_obj = datetime.strptime('September 14, 2109', "%B %d, %Y").date()
    print(d_obj.year)

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