<?php
session_start();

?>
<!DOCTYPE html>
<html>
  <head>
          <meta charset="utf-8" />
          <title>Kino Bilecik</title>
          <link rel="stylesheet" href="assets/css/login.css"/>
           <link href="https://fonts.googleapis.com/css?family=Lato:400,900|Raleway:400,900|Shadows+Into+Light|Poppins:wght@300;500;600&amp;subset=latin-ext" rel="stylesheet" />
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">

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
            <li><a href=<?php if (!isset($_SESSION['user'])) {
              print 'logowanie.php>Zaloguj się</a></li>';
            } elseif (isset($_SESSION['isAdmin'])) {
              print 'panel.php>Panel Administratora</a></li>';
            } else {
              print 'panel.php>Panel Użytkownika</a></li>';} ?>
          </ul>
        </nav>
     </header> 

     <!-- literally the only purpose of this div is to stop background from tiling. wtf -->
      <div class="elstupido" style=" width: 100%; height: 100vmin;">
   
    <div class="background">
        <div class="shape"></div>
        <div class="shape"></div>
    </div>
    <form method="post" action="">
        <h3>Login Here</h3>

        <label for="username">Username</label>
        <input type="text" placeholder="Email or Phone" id="username" name="user">

        <label for="password">Password</label>
        <input type="password" placeholder="Password" id="password" name="pass">

        <button>Log In</button>
        <div class="social">
          <div class="go"><i class="fab fa-google"></i>  Google</div>
          <div class="fb"><i class="fab fa-facebook"></i>  Facebook</div>
        </div>
    </form>
<?php
$user = isset($_POST['user']) ? substr(strip_tags($_POST['user']),0,64) : '';
$pass = isset($_POST['pass']) ? $_POST['pass'] : '';
if(!empty($user) && !empty($pass))

  {
 

  $pass = md5($pass);
  $db = new PDO('sqlite:users.db');
  $sql = "SELECT * FROM users WHERE user=:user AND pass=:pass";
  $res = $db->prepare($sql);
  $res->bindValue(':user', $user);
  $res->bindValue(':pass', $pass);
  $res->execute();
  if($row=$res->fetch())
    {
        $_SESSION['user'] = $row['user'];
        $_SESSION['priv'] = $row['priv'];
    $_SESSION['isAdmin'] = $row['isAdmin'];
        if(($_SESSION['priv'] & 1) > 0) {
      header("Location:panel.php");
        }
    }
  else
    {
    print '<div id = "baderror" style="display: none;">wrong</div>'.PHP_EOL;
    unset($_SESSION['priv'], $_SESSION['user']);
    }
  }
 ?> 
 <script>
  var noidea = document.getElementById("baderror");
  var iwillthinkofabetternamelater = noidea.textContent;
   
  
  if (iwillthinkofabetternamelater.toString == "wrong") {
    window.alert("You fucking donkey")
  }
//TODO
//pop up when bad password
  </script>

      </body>
  
</html>