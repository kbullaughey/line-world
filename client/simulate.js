var socket = new WebSocket("ws://localhost:2601", "lineworld");

var command = "propose";
var timer = setInterval(function() {
  socket.send(command);
}, 100);

socket.onmessage = function(e) {
  if (e.data.match(/^>/)) {
    $('#scene').html(e.data);
  } else {
    $('#message').html(e.data);
  }
};

socket.onclose = function(e) {
  clearTimeout(timer);
};
