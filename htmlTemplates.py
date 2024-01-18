css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e;
    flex-direction: row
}
.chat-message.bot {
    background-color: #475063;
    flex-direction: row-reverse
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  width: 78px;
  height: 78px;
  border-radius: 100%;
  object-fit: cover;
}
.chat-message .message {
  width: 100%;
  padding: 0 1.5rem;
  color: #fff;
}
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://logos-world.net/wp-content/uploads/2020/07/Accenture-Logo-700x394.png"> 
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://i.ibb.co/CQ50xgd/default.png">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''

