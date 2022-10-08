"""
This module contains utility functions that process the data extracted
from the emails into a format more suitable for use with machine
learning algorithms.
"""
import pandas as pd
import numpy as np
import re

from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

import random
random.seed(1746)

import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)


"""
Text Preprocessing
"""
def check_empty(input_string):
    """
    Check if input is considered an empty email.

    An empty email would contain only whitespace.
    
    The implementation simply uses python's strip() to see if
    anything remains after stripping the leading and trailing
    whitespace, which in this case should be an empty string.

    Parameters
    ----------
    input_string : str
        The string to be checked.

    Returns
    -------
    bool
        True if the input is found empty.
    """
    if(input_string.strip()):
        return False
    else:
        return True
        
    
def replace_email(input_string, replacement_string='<emailaddress>'):
    """
    Replace email addresses in a string with a defined string.

    A simple regex is used to search a string for usual patterns
    of email addresses.
    
    The pattern is good enough for the purposes of this project,
    but it should not be considered perfect.
    
    Parameters
    ----------
    input_string : str
        The string that will be searched for email addresses.
        
    replacement_string: str, default '<emailaddress>'
        The string that will replace email addresses.

    Returns
    -------
    str
        The input string with replaced email addresses.
    """
    output_string = re.sub(r'\b[A-Za-z0-9._%+-]+@([A-Za-z0-9.-]+\.[A-Z|a-z]{2,})\b', '<emailaddress>', input_string)

    return output_string

def replace_url(input_string, replacement_string='<urladdress>'):
    """
    Replace URLs in a string with a defined string.

    A regex is used to search a string for usual patterns
    of email addresses. It tries to detect as many URL formats
    as possible, like example.com or https://www.example.com
    and even when there are added parameters, like
    http://example.com/subpage?param1=1&param2=2#service
    This can result in some false positives.
    
    The biggest weakness is that it detects domains of email
    addresses (for example it will replace example.com
    in user@example.com), but this can be solved by replacing
    the emails first.
    
    In order to avoid detecting strings such as document.pdf,
    it uses a list of TLDs from IANA to distinguish those
    from the file extensions. This also means that the regex
    is unable to match IP addresses.
    
    Parameters
    ----------
    input_string : str
        The string that will be searched for URLs.
        
    replacement_string: str, default '<urladdress>'
        The string that will replace URLs.

    Returns
    -------
    str
        The input string with replaced URLs.
        
    See Also
    --------
    replace_email : Replace email addresses in a string with a defined string.
    """
    output_string = re.sub(r'((http|ftp|https)\:\/\/)?([\w_\.-]+(?:\.(zw|zuerich|zone|zm|zip|zero|zara|zappos|za|yun|yt|youtube|you|yokohama|yoga|yodobashi|ye|yandex|yamaxun|yahoo|yachts|xyz|xxx|xn--zfr164b|xn--ygbi2ammx|xn--yfro4i67o|xn--y9a3aq|xn--xkc2dl3a5ee0h|xn--xkc2al3hye2a|xn--xhq521b|xn--wgbl6a|xn--wgbh1c|xn--w4rs40l|xn--w4r85el8fhu5dnra|xn--vuq861b|xn--vhquv|xn--vermgensberatung-pwb|xn--vermgensberater-ctb|xn--unup4y|xn--tiq49xqyj|xn--tckwe|xn--t60b56a|xn--ses554g|xn--s9brj9c|xn--rvc1e0am3e|xn--rovu88b|xn--rhqv96g|xn--qxam|xn--qxa6a|xn--qcka1pmc|xn--q9jyb4c|xn--q7ce6a|xn--pssy2u|xn--pgbs0dh|xn--p1ai|xn--p1acf|xn--otu796d|xn--ogbpf8fl|xn--o3cw4h|xn--nyqy26a|xn--nqv7fs00ema|xn--nqv7f|xn--node|xn--ngbrx|xn--ngbe9e0a|xn--ngbc5azd|xn--mxtq1m|xn--mk1bu44c|xn--mix891f|xn--mgbx4cd0ab|xn--mgbtx2b|xn--mgbt3dhd|xn--mgbpl2fh|xn--mgbi4ecexp|xn--mgbgu82a|xn--mgberp4a5d4ar|xn--mgbcpq6gpa1a|xn--mgbca7dzdo|xn--mgbc0a9azcg|xn--mgbbh1a71e|xn--mgbbh1a|xn--mgbayh7gpa|xn--mgbai9azgqp6j|xn--mgbah1a3hjkrd|xn--mgbab2bd|xn--mgbaam7a8h|xn--mgbaakc7dvf|xn--mgba7c0bbn0a|xn--mgba3a4f16a|xn--mgba3a3ejt|xn--mgb9awbf|xn--lgbbat1ad8j|xn--l1acc|xn--kput3i|xn--kpry57d|xn--kprw13d|xn--kcrx77d1x4a|xn--jvr189m|xn--jlq61u9w7b|xn--jlq480n2rg|xn--j6w193g|xn--j1amh|xn--j1aef|xn--io0a7i|xn--imr513n|xn--i1b6b1a6a2e|xn--hxt814e|xn--h2brj9c8c|xn--h2brj9c|xn--h2breg3eve|xn--gk3at1e|xn--gecrj9c|xn--gckr3f0f|xn--g2xx48c|xn--fzys8d69uvgm|xn--fzc2c9e2c|xn--fpcrj9c3d|xn--flw351e|xn--fjq720a|xn--fiqz9s|xn--fiqs8s|xn--fiq64b|xn--fiq228c5hs|xn--fhbei|xn--fct429k|xn--efvy88h|xn--eckvdtc9d|xn--e1a4c|xn--d1alf|xn--d1acj3b|xn--czru2d|xn--czrs0t|xn--czr694b|xn--clchc0ea0b2g2a9gcd|xn--cg4bki|xn--cckwcxetd|xn--cck2b3b|xn--c2br7g|xn--c1avg|xn--bck1b9a5dre4c|xn--b4w605ferd|xn--9krt00a|xn--9et52u|xn--9dbq2a|xn--90ais|xn--90ae|xn--90a3ac|xn--8y0a063a|xn--80aswg|xn--80asehdb|xn--80aqecdr1a|xn--80ao21a|xn--80adxhks|xn--6qq986b3xl|xn--6frz82g|xn--5tzm5g|xn--5su34j936bgsg|xn--55qx5d|xn--55qw42g|xn--54b7fta0cc|xn--4gbrim|xn--4dbrk0ce|xn--45q11c|xn--45brj9c|xn--45br5cyl|xn--42c2d9a|xn--3pxu8k|xn--3hcrj9c|xn--3e0b707e|xn--3ds443g|xn--3bst00m|xn--30rr7y|xn--2scrj9c|xn--1qqw23a|xn--1ck2e1b|xn--11b4c3d|xin|xihuan|xfinity|xerox|xbox|wtf|wtc|ws|wow|world|works|work|woodside|wolterskluwer|wme|winners|wine|windows|win|williamhill|wiki|wien|whoswho|wf|weir|weibo|wedding|wed|website|weber|webcam|weatherchannel|weather|watches|watch|wanggou|wang|walter|walmart|wales|vuelos|vu|voyage|voto|voting|vote|volvo|volkswagen|vodka|vn|vlaanderen|vivo|viva|vision|visa|virgin|vip|vin|villas|viking|vig|video|viajes|vi|vg|vet|versicherung|verisign|ventures|vegas|ve|vc|vanguard|vana|vacations|va|uz|uy|us|ups|uol|uno|university|unicom|uk|ug|ubs|ubank|ua|tz|tw|tvs|tv|tushu|tunes|tui|tube|tt|trv|trust|travelersinsurance|travelers|travelchannel|travel|training|trading|trade|tr|toys|toyota|town|tours|total|toshiba|toray|top|tools|tokyo|today|to|tn|tmall|tm|tl|tkmaxx|tk|tjx|tjmaxx|tj|tirol|tires|tips|tiffany|tienda|tickets|tiaa|theatre|theater|thd|th|tg|tf|teva|tennis|temasek|tel|technology|tech|team|tdk|td|tci|tc|taxi|tax|tattoo|tatar|tatamotors|target|taobao|talk|taipei|tab|sz|systems|sydney|sy|sx|swiss|swatch|sv|suzuki|surgery|surf|support|supply|supplies|sucks|su|style|study|studio|stream|store|storage|stockholm|stcgroup|stc|statefarm|statebank|star|staples|stada|st|ss|srl|sr|spot|sport|space|spa|soy|sony|song|solutions|solar|sohu|software|softbank|social|soccer|so|sncf|sn|smile|smart|sm|sling|sl|skype|sky|skin|ski|sk|sj|site|singles|sina|silk|si|showtime|show|shouji|shopping|shop|shoes|shiksha|shia|shell|shaw|sharp|shangrila|sh|sg|sfr|sexy|sex|sew|seven|ses|services|sener|select|seek|security|secure|seat|search|se|sd|scot|science|schwarz|schule|school|scholarships|schmidt|schaeffler|scb|sca|sc|sbs|sbi|sb|saxo|save|sas|sarl|sap|sanofi|sandvikcoromant|sandvik|samsung|samsclub|salon|sale|sakura|safety|safe|saarland|sa|ryukyu|rwe|rw|run|ruhr|rugby|ru|rsvp|rs|room|rogers|rodeo|rocks|rocher|ro|rip|rio|ril|ricoh|richardli|rich|rexroth|reviews|review|restaurant|rest|republican|report|repair|rentals|rent|ren|reliance|reit|reisen|reise|rehab|redumbrella|redstone|red|recipes|realty|realtor|realestate|read|re|radio|racing|quest|quebec|qpon|qa|py|pwc|pw|pub|pt|ps|prudential|pru|protection|property|properties|promo|progressive|prof|productions|prod|pro|prime|press|praxi|pramerica|pr|post|porn|politie|poker|pohl|pnc|pn|pm|plus|plumbing|playstation|play|place|pl|pk|pizza|pioneer|pink|ping|pin|pid|pictures|pictet|pics|physio|photos|photography|photo|phone|philips|phd|pharmacy|ph|pg|pfizer|pf|pet|pe|pccw|pay|passagens|party|parts|partners|pars|paris|panasonic|page|pa|ovh|ott|otsuka|osaka|origins|organic|org|orange|oracle|open|ooo|online|onl|ong|one|omega|om|ollo|oldnavy|olayangroup|olayan|okinawa|office|observer|obi|nz|nyc|nu|ntt|nrw|nra|nr|np|nowtv|nowruz|now|norton|northwesternmutual|nokia|no|nl|nissay|nissan|ninja|nikon|nike|nico|ni|nhk|ngo|ng|nfl|nf|nexus|nextdirect|next|news|new|neustar|network|netflix|netbank|net|nec|ne|nc|nba|navy|natura|name|nagoya|nab|na|mz|my|mx|mw|mv|mutual|music|museum|mu|mtr|mtn|mt|msd|ms|mr|mq|mp|movie|mov|motorcycles|moto|moscow|mortgage|mormon|monster|money|monash|mom|moi|moe|moda|mobile|mobi|mo|mn|mma|mm|mls|mlb|ml|mk|mitsubishi|mit|mint|mini|mil|microsoft|miami|mh|mg|merckmsd|menu|men|memorial|meme|melbourne|meet|media|med|me|md|mckinsey|mc|mba|mattel|maserati|marshalls|marriott|markets|marketing|market|map|mango|management|man|makeup|maison|maif|madrid|macys|ma|ly|lv|luxury|luxe|lundbeck|lu|ltda|ltd|lt|ls|lr|lplfinancial|lpl|love|lotto|lotte|london|lol|loft|locus|locker|loans|loan|llp|llc|lk|living|live|lipsy|link|linde|lincoln|limo|limited|lilly|like|lighting|lifestyle|lifeinsurance|life|lidl|li|lgbt|lexus|lego|legal|lefrak|leclerc|lease|lds|lc|lb|lawyer|law|latrobe|latino|lat|lasalle|lanxess|landrover|land|lancia|lancaster|lamer|lamborghini|lacaixa|la|kz|kyoto|ky|kw|kuokgroup|kred|krd|kr|kpn|kpmg|kp|kosher|komatsu|koeln|kn|km|kiwi|kitchen|kindle|kinder|kim|kids|kia|ki|kh|kg|kfh|kerryproperties|kerrylogistics|kerryhotels|ke|kddi|kaufen|juniper|juegos|jprs|jpmorgan|jp|joy|jot|joburg|jobs|jo|jnj|jmp|jm|jll|jio|jewelry|jetzt|jeep|je|jcb|java|jaguar|itv|itau|it|istanbul|ist|ismaili|is|irish|ir|iq|ipiranga|io|investments|intuit|international|int|insure|insurance|institute|ink|ing|info|infiniti|industries|inc|in|immobilien|immo|imdb|imamat|im|il|ikano|ifm|ieee|ie|id|icu|ice|icbc|ibm|hyundai|hyatt|hughes|hu|ht|hsbc|hr|how|house|hotmail|hotels|hoteles|hot|hosting|host|hospital|horse|honda|homesense|homes|homegoods|homedepot|holiday|holdings|hockey|hn|hm|hkt|hk|hiv|hitachi|hisamitsu|hiphop|hgtv|hermes|here|helsinki|help|healthcare|health|hdfcbank|hdfc|hbo|haus|hangout|hamburg|hair|gy|gw|guru|guitars|guide|guge|gucci|guardian|gu|gt|gs|group|grocery|gripe|green|gratis|graphics|grainger|gr|gq|gp|gov|got|gop|google|goog|goodyear|goo|golf|goldpoint|gold|godaddy|gn|gmx|gmo|gmbh|gmail|gm|globo|global|gle|glass|gl|giving|gives|gifts|gift|gi|gh|ggee|gg|gf|george|genting|gent|gea|ge|gdn|gd|gbiz|gb|gay|garden|gap|games|game|gallup|gallo|gallery|gal|ga|fyi|futbol|furniture|fund|fun|fujitsu|ftr|frontier|frontdoor|frogans|frl|fresenius|free|fr|fox|foundation|forum|forsale|forex|ford|football|foodnetwork|food|foo|fo|fm|fly|flowers|florist|flir|flights|flickr|fk|fj|fitness|fit|fishing|fish|firmdale|firestone|fire|financial|finance|final|film|fido|fidelity|fiat|fi|ferrero|ferrari|feedback|fedex|fast|fashion|farmers|farm|fans|fan|family|faith|fairwinds|fail|fage|extraspace|express|exposed|expert|exchange|events|eus|eurovision|eu|etisalat|et|estate|esq|es|erni|ericsson|er|equipment|epson|enterprises|engineering|engineer|energy|emerck|email|eg|ee|education|edu|edeka|eco|ec|eat|earth|dz|dvr|dvag|durban|dupont|dunlop|dubai|dtv|drive|download|dot|domains|dog|doctor|docs|do|dnp|dm|dk|dj|diy|dish|discover|discount|directory|direct|digital|diet|diamonds|dhl|dev|design|desi|dentist|dental|democrat|delta|deloitte|dell|delivery|degree|deals|dealer|deal|de|dds|dclk|day|datsun|dating|date|data|dance|dad|dabur|cz|cyou|cymru|cy|cx|cw|cv|cuisinella|cu|cruises|cruise|crs|crown|cricket|creditunion|creditcard|credit|cr|cpa|courses|coupons|coupon|country|corsica|coop|cool|cookingchannel|cooking|contractors|contact|consulting|construction|condos|comsec|computer|compare|company|community|commbank|comcast|com|cologne|college|coffee|codes|coach|co|cn|cm|clubmed|club|cloud|clothing|clinique|clinic|click|cleaning|claims|cl|ck|cityeats|city|citic|citi|citadel|cisco|circle|cipriani|ci|church|chrome|christmas|chintai|cheap|chat|chase|charity|channel|chanel|ch|cg|cfd|cfa|cf|cern|ceo|center|cd|cc|cbs|cbre|cbn|cba|catholic|catering|cat|casino|cash|case|casa|cars|careers|career|care|cards|caravan|car|capitalone|capital|capetown|canon|cancerresearch|camp|camera|cam|calvinklein|call|cal|cafe|cab|ca|bzh|bz|by|bw|bv|buzz|buy|business|builders|build|bugatti|bt|bs|brussels|brother|broker|broadway|bridgestone|bradesco|br|box|boutique|bot|boston|bostik|bosch|booking|book|boo|bond|bom|bofa|boehringer|boats|bo|bnpparibas|bn|bmw|bms|bm|blue|bloomberg|blog|blockbuster|blackfriday|black|bj|biz|bio|bingo|bing|bike|bid|bible|bi|bharti|bh|bg|bf|bet|bestbuy|best|berlin|bentley|beer|beauty|beats|be|bd|bcn|bcg|bbva|bbt|bbc|bb|bayern|bauhaus|basketball|baseball|bargains|barefoot|barclays|barclaycard|barcelona|bar|bank|band|bananarepublic|banamex|baidu|baby|ba|azure|az|axa|ax|aws|aw|avianca|autos|auto|author|auspost|audio|audible|audi|auction|au|attorney|athleta|at|associates|asia|asda|as|arte|art|arpa|army|archi|aramco|arab|ar|aquarelle|aq|apple|app|apartments|aol|ao|anz|anquan|android|analytics|amsterdam|amica|amfam|amex|americanfamily|americanexpress|amazon|am|alstom|alsace|ally|allstate|allfinanz|alipay|alibaba|alfaromeo|al|akdn|airtel|airforce|airbus|aig|ai|agency|agakhan|ag|africa|afl|af|aetna|aero|aeg|ae|adult|ads|adac|ad|actor|aco|accountants|accountant|accenture|academy|ac|abudhabi|abogado|able|abc|abbvie|abbott|abb|abarth|aarp|aaa)\b))([\w.,?^=%&@:/~+#-]*[\w?^=%&@/~+#-])?', replacement_string, input_string)

    return output_string


def tokenize(input_text):
    """
    Convert a string of text to a list with words.
    
    The initial NLTK tokenization produces tokens with
    punctuation marks or other special characters.
    A regex filter is then applied to throw out all such
    tokens, keeping only alphanumeric characters (a-z and 0-9),
    underscores (_) and dashes (-), so that tokens like "e-mail"
    are not removed.
    
    Also, there is a conversion to lowercase.
    
    Parameters
    ----------
    input_text : str
        The string that will be tokenized.
    
    Returns
    -------
    list of str
        The list that contains the cleaned-up tokens.
    """
    lowercase = input_text.lower()
    token_list = nltk.word_tokenize(lowercase)
    clean_list = [word for word in token_list if len(re.findall(r'^[a-zA-Z0-9]+-?[\w-]*$', word)) == 1]
    
    return clean_list

def remove_stopwords(tokenized_text):
    """
    Remove stopwords from a list of words.
        
    Parameters
    ----------
    tokenized_text : list
        The list of words to process.
    
    Returns
    -------
    list
        The word list without the stopwords.
    """
    stop_words = stopwords.words('english')
    clean_list = [word for word in tokenized_text if word not in stop_words]
    
    return clean_list


def get_wordnet_pos(treebank_tag):
    """
    Converts a Treebank part-of-speech tag to WordNet.
    
    Parameters
    ----------
    treebank_tag : str
        The Treebank POS tag.
    
    Returns
    -------
    str or None
        A wordnet POS constant or None if no maping was found.
    """
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def lemmatize(token_list):
    """
    Lemmatizes a list of tokens using WordNet lemmatizer.
    
    Depending on the result of get_wordnet_pos(), the function
    will either use a POS tag to improve lemmatization or not.
    
    Parameters
    ----------
    token_list : list of str
        The list that contains the tokenized words.
    
    Returns
    -------
    list of str
        A list with the lemmas of the input words.
        
    See Also
    --------
    get_wordnet_pos : Converts a Treebank part-of-speech tag to WordNet.
    """
    lemmatizer = WordNetLemmatizer()
    tagged_list = nltk.pos_tag(token_list)
    
    lemmatized_list = [lemmatizer.lemmatize(word)
                       if get_wordnet_pos(tag) is None
                       else lemmatizer.lemmatize(word, pos=get_wordnet_pos(tag))
                       for word, tag in tagged_list]
    
    return lemmatized_list


def sanitize_whitespace(input_string):
    """
    Remove some extra whitespace and fix stray dots.

    First, check if a line in the input only contains a dot, in which
    case the dot is appended to the previous line that contains text
    and the line with the dot is removed.
    
    Finally, strip leading and trailing whitespace from the output.

    Parameters
    ----------
    input_string : str
        The string to be sanitized.

    Returns
    -------
    str
        The sanitized string.
    """
    lines = input_string.split('\n')
    for index, line  in enumerate(lines):
        if line=='.':
            lines.remove(line)
            for i in range(1, len(lines)): # check for lines that contain text
                if (lines[index-i] != ''):
                    lines[index-i] += '.'
                    break
                    
    output_string = '\n'.join(lines)
    
    return output_string.strip()

def sanitize_addresses(input_string):
    """
    Remove extra special characters from addresses.

    Parameters
    ----------
    input_string : str
        The string to be sanitized.

    Returns
    -------
    str
        The sanitized string.
    """
    less = re.sub(r'<+', '<', input_string)
    more = re.sub(r'>+', '>', less)
    
    return more


"""
Dataset Processing
"""
def dataset_split(dataset, percent=0.2, keep_index=False):
    """
    Split a dataset into train and test sets.
    
    It is a wrapper for sklearn's train_test_split() that also resets
    the index.
    
    Parameters
    ----------
    dataset : pandas.DataFrame
        The dataset to split.
    percent : int, default 20
        If it is in range [0, 100], it means the percentage of the
        original dataset to use as the test set.
    keep_index : bool, default False
        If True, keep the old index in a new column, otherwise drop it.
        
    Returns
    -------
    train : pandas.DataFrame
        The result train set.
    test : pandas.DataFrame
        The result test set.
    """
    train, test = train_test_split(dataset, test_size=percent/100, random_state=1746)
    
    if not keep_index:
        train = train.reset_index(drop=True)
        test = test.reset_index(drop=True)
    else:
        train = train.reset_index()
        test = test.reset_index()
    
    return train, test

def dataset_add_columns(dataset, columns, column_names, position=0):
    """
    Add columns to a dataset.
    
    This is mostly used to streamline the addition of class and id
    columns to datasets that contain only features.
    
    The extra columns will appear in the reverse order from their
    order on the input lists, and on the beginning of the DataFrame
    by default.
    
    Parameters
    ----------
    dataset : pandas.DataFrame
        The dataset to add the columns.
    columns : list of pandas.Series
        The values of the columns to be added.
    column_names : list of str
        The names of the columns to be added.
    position : int, default 0
        The position on the column index to put the new columns.
        The default is at the beginning (position 0).
        
    Returns
    -------
    pandas.DataFrame
        The initial DataFrame with the added columns.
        
    Raises
    ------
    ValueError
        If columns and column_names are not the same size.
    """
    if len(columns) != len(column_names):
        raise ValueError("The lists of names and values have to be the same size.")
    else:
        for col, name in zip(columns, column_names):
            dataset.insert(position, name, col)
    
    return dataset


def separate_features_target(dataframe, num_cols_ignore=2, class_col_name='email_class'):
    """
    Separate feature columns from the target column.
    
    It assumes that any non-feature columns are in the
    beginning of the dataset (right after the index).
    
    Essentially it is the inverse of a specific case of
    dataset_add_columns().
    
    Parameters
    ----------
    dataframe : pandas.DataFrame
        The DataFrame with data to split.
    num_cols_ignore : int
        The number of non-feature columns to skip.
    class_col_name : str
        The name of the target column.
        
    Returns
    -------
    dict
    {'features': pandas.DataFrame,
     'target': pandas.Series}
        A dictionary containing the features and target.
    """
    return {'features': dataframe[dataframe.columns[num_cols_ignore:]],
            'target': dataframe[class_col_name]}

"""
Missing Values
"""
def impute_missing(train, test, strategy='mean'):
    """
    Fills the missing values of a dataset using the provided strategy.
    
    The sklearn.SimpleImputer instance will be fitted on the train
    data and will be used to transform the test data too.
    
    Parameters
    ----------
    train : pandas.DataFrame
        The train set to process.
    test : pandas.DataFrame
        The test set to process.
    strategy : {'mean', 'median', 'most_frequent'}
        This parameter will be passed to SimpleImputer to determine
        the imputation method.
        
    Returns
    -------
    train : pandas.DataFrame
        The imputed training set.
    test : pandas.DataFrame
        The imputed test set.
    """
    imp_mean = SimpleImputer(missing_values=np.NaN, strategy=strategy)
    
    train = pd.DataFrame(imp_mean.fit_transform(train), columns=train.columns)
    test = pd.DataFrame(imp_mean.transform(test), columns=train.columns)
    
    return train, test