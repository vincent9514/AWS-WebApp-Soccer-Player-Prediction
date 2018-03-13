FIFA Application Tutorial
=============================================

This is the tutorial to set up the App locally

* Step 0: Create an AWS account:

  Note: AWS requires a credit card for registration. But our example will exist entirely on the AWS Free Tier, so you won’t be charged.

* Step 1: Download the application code from github repository:

  Fork my Github repo: https://github.com/vincent9514/MSiA423-Soccer-Player-Prediction.git

  We’re going to deploy a simple app that reads and writes from a database using Flask-SQLAchemy. Dig into the code if you’d like — it should run locally if you run python application.py (after you set you the environment in Step 4).

* Step 2: Set up your Flask environment:

  In the directory where the example code exists, create a Python virtual environment.

  $ virtualenv flask-aws

  $ source flask-aws/bin/activate

  Then install the packages needed for this demo with:

  $ pip install -r requirements.txt

* Step 3: Create a MySQL database using AWS RDS
  
  On the AWS console, go to Services > RDS.
 
  Next, click “Launch a DB Instance”

  Select “MySql Community Edition”

  Select “No” for multi-AZ deployment — this will keep us in the Free Tier.

  Select “DB Instance Class” as db.t2.micro (keeps us in the Free Tier), “Multi-AZ Deployment” as “no” (they’re really pushing that, right?), and set up your DB instance name, user name, and password. 

  For the advanced DB settings, leave the security group is “default” and set the Database Name to whatever you’d like.   

  Click “Launch DB Instance” then “View DB Instances.” 

* Step 3.5: Modify the permissions on your DB  

  Go to your AWS dashboard, click “EC2” and you’ll see the screen below. Click “Security Groups”:

  Click “Create a Security Group.” Now you can modify who can access your DB. 

  Scroll down to “Network and Security” and change it to the security group we just created.

* Step 4: Add tables to your DB instance
  
  First, go to your AWS console, RDS, and click on “DB Instances” Copy the “Endpoint” string — this is the URL to your AWS DB:  

  Edit the config.py file to include the username, password, and db name you entered earlier, in the format:

  SQLALCHEMY_DATABASE_URI = ‘mysql+pymysql://<db_user>:<db_password>@<endpoint>/<db_url>’

  Now create the tables in your (currently) empty database by running

  $ python db_create.py

* Step 4.5 Test the Flask App:

  $ python application.py

* Step 5: Set up Elastic Beanstalk Environment

  $ pip install awsebcli

  To create a new user, go to the AWS Console and select Identity and Access Management

  Initialize our Elastic Beanstalk environment

    $ eb init

  You’ll see:

  Select a default region
  1) us-east-1 : US East (N. Virginia)
  2) us-west-1 : US West (N. California)
  3) us-west-2 : US West (Oregon)
  4) eu-west-1 : EU (Ireland)
  5) eu-central-1 : EU (Frankfurt)
  6) ap-southeast-1 : Asia Pacific (Singapore)
  7) ap-southeast-2 : Asia Pacific (Sydney)
  8) ap-northeast-1 : Asia Pacific (Tokyo)
  9) sa-east-1 : South America (Sao Paulo)
  (default is 3): 1

  Chose the location closest to you (mine is Northern Virginia). Next you’ll be prompted for the AWS ID and Secret Key for the user “flaskdemo” you saved somewhere:

  You have not yet set up your credentials or your credentials are incorrect
  You must provide your credentials.
  (aws-access-id): <enter the 20 digit AWS ID>
  (aws-secret-key): <enter the 40 digit AWS secret key>

  Next you’ll see:

  Select an application to use
  1) [ Create new Application ]
  (default is 1): 1

  Next we create the environment name. Hit “Enter” to use the default values:

  Enter Application Name
  (default is “flask-aws-tutorial”):
  Application flask-aws-tutorial has been created.

  Now the EBCLI just wants to make sure we’re using Python:

  It appears you are using Python. Is this correct?
  (y/n): y

  Select your Python version. I’m a fan of 2.7 and wrote this example using it, so users of 3+ may find some incompatibilities with this code.

  Select a platform version.
  1) Python 3.4
  2) Python 2.7
  3) Python
  4) Python 3.4 (Preconfigured — Docker)
  (default is 1): 1

  You have the option of creating an SSH connection to this instance. We won’t need to use it, so I recommend “no.” (If you need to ssh into this instance later, you can change the preferences of your EC2 instance from the AWS console later.)

  Do you want to set up SSH for your instances?
  (y/n): n

  Okay, now we’re all set up.

* Step 6: Deploy our Flask Application

  $ eb create

  Now we have to create an environment name and DNS CNAME for our app. 

  Once you’ve selected a unique DNS CNAME, you’ll see status updates as the app is deployed.

  When the uploading finishes, you’ll see:

  INFO: Application available at thisisacoolflaskapp.elasticbeanstalk.com.
  INFO: Successfully launched environment: flask-aws-tutorial-dev

  Point your web browser to that URL and you’ll see your Flask app live! 

* Step 7: Check out the app

  Whenever you update a file, simply type

  $ eb deploy

  when your new changes are ready. 

* Congrats on your first AWS site














































