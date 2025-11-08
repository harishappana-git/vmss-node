terraform {
  backend "s3" {
    bucket         = "CHANGEME-tfstate-bucket"
    key            = "vmss-node/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "CHANGEME-tfstate-lock"
    encrypt        = true
  }
}
