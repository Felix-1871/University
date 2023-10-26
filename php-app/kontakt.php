<?php
session_start();
?>
<!DOCTYPE html>
<html>
  <head>
          <meta charset="utf-8" />
          <title>Kino Bilecik</title>
<link rel="stylesheet" href="assets/css/fontello/fontello.css" />           <link rel="stylesheet" href="assets/css/contact.css">
           <link href="https://fonts.googleapis.com/css?family=Lato:400,900|Raleway:400,900|Shadows+Into+Light&amp;subset=latin-ext" rel="stylesheet" />

          </head>
  <body>
     <header>
        <h1 class="logo"
><i class="icon-video-2"></i> Kino Bilecik</h1>

        <nav id="topnav">
          <ul class="menu">
            <li><a href="index.php">Strona główna</a></li>
            <li><a href="repertuar.php">Repertuar</a></li>
            
            
            <li><a href="kontakt.php">Kontakt</a></li>
            <li><a href="<?php if (!isset($_SESSION['user'])) {
              print 'logowanie.php">Zaloguj się</a></li>';
            } elseif (isset($_SESSION['isAdmin'])) {
              print 'panel.php">Panel Administratora</a></li>';
            } else {
              print 'panel.php">Panel Użytkownika</a></li>';} ?>
            
          </ul>
        </nav>
     </header>

      <div id="text"><div id="text">
  <div class="line">
    <p class="word">The very best</p>
  </div>

  <div class="line">
    <p class="word">Femboy</p>
    <p class="word">&</p>
  </div>

  <div class="line">
    <p class="word">Programmer</p>
  </div>
  
  <div class="line">
    <a 
       id="channel-link" 
       href="https://github.com/Felix-1871" 
       target="_blank" 
       class="word fancy"
    >
      @Feli_x
    </a>
  </div>
</div>
<script>
  const rand = (min, max) => Math.floor(Math.random() * (max - min + 1)) + min;

const enhance = id => {
  const element = document.getElementById(id),
        text = element.innerText.split("");
  
  element.innerText = "";
  
  text.forEach((value, index) => {
    const outer = document.createElement("span");
    
    outer.className = "outer";
    
    const inner = document.createElement("span");
    
    inner.className = "inner";
    
    inner.style.animationDelay = `${rand(-5000, 0)}ms`;
    
    const letter = document.createElement("span");
    
    letter.className = "letter";
    
    letter.innerText = value;
    
    letter.style.animationDelay = `${index * 1000 }ms`;
    
    inner.appendChild(letter);    
    
    outer.appendChild(inner);    
    
    element.appendChild(outer);
  });
}

enhance("channel-link");
</script>




    </footer>
  </body>

</html>