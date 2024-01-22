css = '''
<style>
.chat-container {
    display: flex;
    flex-direction: column;
    max-width: 800px;
    margin: auto;
}

.chat-message {
    padding: 1rem;
    border-radius: 1rem;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); 
    transition: all 0.3s ease; 
    font-size: 1rem; 
}

.chat-message.user {
    background-color: #2b313e;  
    color: #ffffff;  
    justify-content: flex-end;
    flex-direction: row-reverse;
    text-align: right;
}

.chat-message.bot {
    background-color: #475063;  
    color: #f8f9fa; 
    justify-content: flex-start;
    text-align: left;
}

.chat-message .avatar {
    flex: 0 0 auto;
    width: 75px;
    height: 75px;
    margin: 0.5rem;
    border-radius: 30%;
    overflow: hidden;
}

.chat-message .avatar img {
    width: 100%;
    height: 100%;
    object-fit: cover;
}

.chat-message .message {
    flex: 1;
    padding: 0 1rem;
    color: #fff;
    word-wrap: break-word;
}

/* Responsive design adjustments */
@media (max-width: 600px) {
    .chat-container {
        width: 100%;
    }
    .chat-message {
        padding: 0.5rem;
    }
    .chat-message .message {
        padding: 0 0.5rem;
    }
}
</style>

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


