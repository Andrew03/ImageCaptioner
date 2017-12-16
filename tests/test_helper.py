def test_equal(obj, expected, function, message, *params):
  recieved = function(*params)
  obj.assertEqual(expected, recieved, message + "\n" + has_conflict(expected, recieved))
  
def has_conflict(expected, recieved):
  return "Expected: " + str(expected) + "\n" + "Recieved: " + str(recieved)
