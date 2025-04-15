package com.gbh.gbh_mm.user.model.request;

import lombok.Getter;

@Getter
public class RequestUpdateSalary {
    private long salary;
    private int date;
    private String account;
}
