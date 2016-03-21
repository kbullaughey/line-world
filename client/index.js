var app = require('express')();
var http = require('http').Server(app);

app.get('/', function(req, res){
  res.sendFile(__dirname + '/client.html');
});

app.get('/play.js', function(req, res){
  res.sendFile(__dirname + '/play.js');
});

http.listen(2600, function(){
  console.log('listening on *:2600');
});
