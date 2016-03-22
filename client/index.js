var app = require('express')();
var http = require('http').Server(app);

app.get('/', function(req, res){
  res.sendFile(__dirname + '/client.html');
});

app.get('/ai', function(req, res){
  res.sendFile(__dirname + '/ai.html');
});

app.get('/play.js', function(req, res){
  res.sendFile(__dirname + '/play.js');
});

app.get('/simulate.js', function(req, res){
  res.sendFile(__dirname + '/simulate.js');
});

http.listen(2600, function(){
  console.log('listening on *:2600');
});
