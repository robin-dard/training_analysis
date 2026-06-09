from parsers.garmin_client import GarminClient

client = GarminClient()
client.sync(from_date="2018-01-01")
