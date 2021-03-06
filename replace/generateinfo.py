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
    seasons = {'spring' : 'January 1', 'summer' : 'April 1', 'autumn' : 'August 1', 'winter' : 'October 1'}
    weekday = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
    normalize_weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    holidays = {"thanksgiving": "15 August", "ramada": "15 May", "easter": "1 April"}
    symbol = ['\'', 'of']
    ages_word = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    ages_number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    day_shift = 5
    date_shift = 5
    month_shift = 3
    year_shift = 10
    season_shif = 1
    health_plan = ['Medicare Cost Plans', 'Demonstrations/Pilot Programs', "Programs of All-inclusive Care for the Elderly (PACE)", 
            "Medication Therapy Management (MTM) programs for complex health needs"]

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
        piece1 = self.generate_username(3, chars = string.digits)
        piece2 = self.generate_username(3, chars = string.digits)
        piece3 = self.generate_username(4, chars = string.digits)
        return piece1 + '-' + piece2 + '-' + piece3

    def generate_email(self):
        return self._faker.ascii_company_email()

    def generate_URL(self):
        return self._faker.url()

    def generate_IDNum(self):
        return self._faker.isbn10(separator="-")

    def generate_BioID(self):
        return self.generate_username(size = 7)

    def generate_device(self):
        return self._faker.isbn10(separator="-")

    def generate_medicalrecord(self):
        return self._faker.isbn13(separator="-")

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

    def generate_street_add(self):
        return self._faker.street_address()


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
            decade = re.split(r'(\d+)', date_process)
            decade[1] = str(int(decade[1]) + 10)
            return  "".join(decade)

        #check date is day:
        week_day = date_process.split()
        if len(week_day) == 1:
            for index, day in enumerate(self.weekday):
                if re.match(day, date_process):
                    return self.normalize_weekdays[index + self.day_shift - len(self.weekday)] if index + self.day_shift >= len(self.weekday) else self.weekday[index + self.day_shift]

        d_obj = None

        h_flag = 'N'
        index_flag = 0
        holiday_year = re.findall(r"[\w']+|[.,!?;\/+]", date_process)
        
        #check holidays and year
        for h_index, h_word in enumerate(holiday_year):
            if h_word in self.holidays.keys():
                h_flag = 'H'
                break

        for s_index, s_word in enumerate(holiday_year):
            if s_word in self.seasons.keys():
                h_flag = 'S'
                break

        #convert holiday to a specific day
        if h_flag == 'H':
            holiday_year[h_index] =  self.holidays[h_word]
            reg_date = " ".join(holiday_year)

        elif h_flag == 'S':
            holiday_year[s_index] = self.seasons[s_word]
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

        return datetime.strptime(str(f_m) + '/' + str(f_d) + '/' + str(f_y), '%m/%d/%Y').strftime(p)


    def generate_age(self, age):
        # if re.match(self.date_decade, age):
        #     return age
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


    def generate_healthplan(self):
        return random.choice(self.health_plan)

    def get_random(self, n, end, start = 1):
        return random.choice(list(range(start, n)) + list(range(n+1, end)))

if __name__ == "__main__":
    fake = FakerInfo()

    print(fake.generate_email())

    # print(fake.generate_age('13 month'))
    # print(fake.generate_age('15\'s'))
    # print(fake.generate_age('15      \'s'))
    # print(fake.generate_age('25s'))
    # print(fake.generate_age('13y7.7m'))
    # print(fake.generate_age('6 weeks'))
    # print(fake.generate_age('6weeks'))
    # print(fake.generate_age('6mos'))
    # print(fake.generate_age('6 mos'))
    # print(fake.generate_age('45'))
    # print(fake.generate_age('5'))
    # print(fake.generate_age('twenty four'))
    # print(fake.generate_age('twenty-four'))
    # print(fake.generate_age('twenty'))
    # print(isinstance('45', int))

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